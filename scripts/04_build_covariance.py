from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from src import candidate_sets as CS
from src import noise as N
from src import util
from src import seeding


@torch.no_grad()
def estimate_g_priv(model, ids, k, prefix_lens=(2, 5, 10, 20), n_top=80, n_rand=20,
                   attention_mask=None, row_mask=None):
    H = M.hidden_size(model)
    G_acc = torch.zeros(H, H)
    n = 0
    for t in prefix_lens:
        if t >= ids.shape[1]: continue
        top = CS.topk_candidates(model, ids, position=t, k_top=n_top)
        rand = CS.random_candidates(
            M.vocab_size(model), ids.shape[0], n_rand, seed=12345 + t,
        ).to(ids.device)
        cand = torch.cat([top, rand], dim=1)
        cloud = CS.candidate_cloud(
            model, ids, position=t, candidate_ids=cand,
            k_split=k, chunk_size=M.cloud_chunk_size(model),
        )
        for b in range(cloud.shape[0]):
            if row_mask is not None and not bool(row_mask[b, t]):
                continue
            X = cloud[b].float() - cloud[b].float().mean(dim=0, keepdim=True)
            n_x = X.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            Xn = X / n_x
            G_acc += (Xn.T @ Xn).cpu(); n += Xn.shape[0]
    return G_acc / max(1, n)


def build_covariances(G_priv, G_util, hidden, sigma0, rank, total_extra_var, rho=1e-4):
    Gp = G_priv.float() + rho * torch.eye(hidden)
    Gu = G_util.float() + rho * torch.eye(hidden)
    L = torch.linalg.cholesky(Gu)
    L_inv = torch.linalg.solve_triangular(L, torch.eye(hidden), upper=False)
    A = L_inv @ Gp @ L_inv.T
    A = (A + A.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(A)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    U_gen = (L_inv.T @ eigvecs)
    U_gen = U_gen / U_gen.norm(dim=0, keepdim=True).clamp(min=1e-12)
    U_top = U_gen[:, :rank]
    Q, _ = torch.linalg.qr(U_top)
    lam_top = eigvals[:rank].clamp(min=0)
    lam_top = lam_top / lam_top.sum().clamp(min=1e-12) * total_extra_var
    return {
        "lowrank_struct": N.GaussianCov(name=f"lrstruct_{sigma0:g}_r{rank}",
                                        hidden=hidden, sigma0=0.0, U=Q, lam=lam_top,
                                        note="low-rank only"),
    }


@torch.no_grad()
def run_one(model_id, k, device="cuda:0", dtype=torch.float32,
            sigma0_fracs=(0.5, 1.5, 3.0, 6.0, 10.0), ranks=(16,),
            task_name="sst2", prefix_lens=None):
    print(f"\n--- covariance build {task_name} {model_id}  k={k} ---", flush=True)
    cache_p = util.art_path(
        "activations", model_id, task_name, f"split_{k}", "cache.pt",
    )
    geom_dir = util.art_path("geometry", model_id, task_name, f"split_{k}")
    if not (geom_dir / "U_T.pt").exists():
        print(f"  no Phase 2 outputs at {geom_dir}, skip"); return None
    cache = torch.load(cache_p, weights_only=False)
    train = cache["train"]
    ids = train["ids"][:32].to(device)
    mask = train["mask"][:32].to(device)
    privacy = train.get("privacy_mask", train["mask"])[:32]
    clean = train["clean_a"][:32]
    import json
    meta_path = util.art_path(
        "activations", model_id, task_name, f"split_{k}", "metadata.json",
    )
    if prefix_lens is None and meta_path.exists():
        prefix_lens = tuple(
            json.loads(meta_path.read_text()).get(
                "privacy_positions", [2, 5, 10, 20],
            )
        )
    if prefix_lens is None:
        prefix_lens = (2, 5, 10, 20)
    H = clean.shape[-1]
    median_norm = float(clean.float().norm(dim=-1).median())
    print(f"  H={H}  median_norm={median_norm:.3f}", flush=True)

    model, _ = M.load_model(model_id, dtype=dtype, device=device)
    Gp = estimate_g_priv(
        model, ids, k=k, prefix_lens=prefix_lens,
        attention_mask=mask, row_mask=privacy,
    )
    U_T = torch.load(geom_dir / "U_T.pt", weights_only=True)
    Gu = (U_T.float() @ U_T.float().T)

    torch.save(Gp, geom_dir / "G_priv.pt")
    torch.save(Gu, geom_dir / "G_util_k.pt")
    util.write_json(geom_dir / "spectra.json", {
        "median_activation_norm": median_norm,
        "G_priv_trace": float(Gp.diag().sum()),
        "G_util_trace": float(Gu.diag().sum()),
    })

    cov_dir = util.art_path(
        "covariances", model_id, task_name, f"split_{k}",
    )
    cov_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sf in sigma0_fracs:
        # sigma0 is per-coordinate stddev; sf is "noise vector norm / signal vector norm"
        # E||eta|| = sigma0 * sqrt(H), so sigma0 = sf * median_norm / sqrt(H)
        sigma0 = sf * median_norm / (H ** 0.5)
        total_extra = (sigma0 ** 2) * H * 2.0
        for r in ranks:
            covs = build_covariances(Gp, Gu, hidden=H, sigma0=sigma0, rank=r,
                                     total_extra_var=total_extra)
            tag = f"sigma0_{sf:g}_r{r}"
            for name, cov in covs.items():
                fn = cov_dir / f"{tag}__{name}.pt"
                torch.save({"U": cov.U, "lam": cov.lam, "sigma0": cov.sigma0,
                            "name": cov.name, "hidden": cov.hidden,
                            "note": cov.note}, fn)
                summaries.append({"tag": tag, "name": name, "sigma0": float(cov.sigma0),
                                  "rank": int(cov.rank), "trace": float(cov.trace())})
    util.write_json(cov_dir / "manifest.json", summaries)
    print(f"  built {len(summaries)} cov files", flush=True)
    del model; torch.cuda.empty_cache()
    return {"model_id": model_id, "k": k, "median_norm": median_norm, "n_cov": len(summaries)}


def _write_covariance_reports(task_name: str, rows: list[dict]) -> None:
    """Write one covariance report per model.

    Args:
        task_name: Benchmark name.
        rows: Summaries returned by ``run_one``.
    """
    models = []
    for row in rows:
        if row["model_id"] not in models:
            models.append(row["model_id"])
    for model_id in models:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        report = [
            f"# Phase 3-4 {task_name} Geometry & Covariance",
            "",
            "| model | k | median norm | # cov files |",
            "|---|---:|---:|---:|",
        ]
        for row in model_rows:
            report.append(
                f"| {row['model_id']} | {row['k']} | "
                f"{row['median_norm']:.2f} | {row['n_cov']} |"
            )
        directory = util.art_path("reports", model_id, task_name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "04_build_covariance.md").write_text("\n".join(report))
        util.write_json(
            directory / "covariance_summary.json", {"rows": model_rows},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    # defend-sc isolation: expose model/k/rank/sf so the rank-8 lowrank_struct base
    # covariance the frontier needs can be built without editing source. Defaults
    # reproduce the original hardcoded behavior (gpt2+Qwen, k in {0,4,8}, rank 16).
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--ranks", nargs="+", type=int, default=[16])
    ap.add_argument("--sfs", nargs="+", type=float,
                    default=[0.5, 1.5, 3.0, 6.0, 10.0])
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeding.set_seed(args.seed)

    rows = []
    for mid in args.models:
        for k in args.ks:
            r = run_one(
                mid, k, dtype=M.default_dtype(mid, args.device),
                device=args.device, sigma0_fracs=tuple(args.sfs),
                ranks=tuple(args.ranks), task_name=args.task,
            )
            if r:
                r["task"] = args.task
                rows.append(r)
    _write_covariance_reports(args.task, rows)


if __name__ == "__main__":
    main()
