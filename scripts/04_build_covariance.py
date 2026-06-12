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
                   attention_mask=None):
    H = M.hidden_size(model)
    G_acc = torch.zeros(H, H)
    n = 0
    for t in prefix_lens:
        if t >= ids.shape[1]: continue
        top = CS.topk_candidates(model, ids, position=t, k_top=n_top)
        rand = CS.random_candidates(model.config.vocab_size, ids.shape[0], n_rand,
                                    seed=12345 + t).to(ids.device)
        cand = torch.cat([top, rand], dim=1)
        cloud = CS.candidate_cloud(model, ids, position=t, candidate_ids=cand,
                                   k_split=k, chunk_size=256)
        for b in range(cloud.shape[0]):
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
            sigma0_fracs=(0.5, 1.5, 3.0, 6.0, 10.0), ranks=(16,)):
    print(f"\n--- covariance build {model_id}  k={k} ---", flush=True)
    safe = model_id.replace("/", "_")
    cache_p = util.ART / "activations" / safe / "sst2" / f"split_{k}" / "cache.pt"
    geom_dir = util.ART / "geometry" / safe / "sst2" / f"split_{k}"
    if not (geom_dir / "U_T.pt").exists():
        print(f"  no Phase 2 outputs at {geom_dir}, skip"); return None
    cache = torch.load(cache_p, weights_only=False)
    train = cache["train"]
    ids = train["ids"][:32].to(device)
    mask = train["mask"][:32].to(device)
    clean = train["clean_a"][:32]
    H = clean.shape[-1]
    median_norm = float(clean.float().norm(dim=-1).median())
    print(f"  H={H}  median_norm={median_norm:.3f}", flush=True)

    model, _ = M.load_model(model_id, dtype=dtype, device=device)
    Gp = estimate_g_priv(model, ids, k=k, attention_mask=mask)
    U_T = torch.load(geom_dir / "U_T.pt", weights_only=True)
    Gu = (U_T.float() @ U_T.float().T)

    torch.save(Gp, geom_dir / "G_priv.pt")
    torch.save(Gu, geom_dir / "G_util_k.pt")
    util.write_json(geom_dir / "spectra.json", {
        "median_activation_norm": median_norm,
        "G_priv_trace": float(Gp.diag().sum()),
        "G_util_trace": float(Gu.diag().sum()),
    })

    cov_dir = util.ART / "covariances" / safe / "sst2" / f"split_{k}"
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeding.set_seed(args.seed)

    rows = []
    for mid in args.models:
        for k in args.ks:
            r = run_one(mid, k, dtype=torch.float32, device=args.device,
                        sigma0_fracs=tuple(args.sfs), ranks=tuple(args.ranks))
            if r: rows.append(r)
    util.write_json(util.ART / "covariance_summary.json", {"rows": rows})
    rep = ["# Phase 3-4 SST-2 Geometry & Covariance", "",
           "| model | k | median norm | # cov files |",
           "|---|---:|---:|---:|"]
    for r in rows:
        rep.append(f"| {r['model_id']} | {r['k']} | {r['median_norm']:.2f} | {r['n_cov']} |")
    (util.ART / "reports" / "04_build_covariance.md").write_text("\n".join(rep))


if __name__ == "__main__":
    main()
