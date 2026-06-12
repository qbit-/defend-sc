"""Phase 2 SST-2: subspace alignment with label-logit T_k."""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from src import split_model as SM
from src import candidate_sets as CS
from src import geometry as G
from src import sst2_geometry as SG
from src import sst2_data as D
from src import util
from src import seeding


@torch.no_grad()
def per_prompt_local_S(model, ids, prefix_lens, k, n_top=80, n_rand=20,
                       energy=0.9, max_rank=24, attention_mask=None):
    """Vocabulary-wide candidate clouds at intermediate prefix positions, same as
    tame_sipit Phase 2 (S is the *prompt-recovery* subspace, not label-discriminative)."""
    out = []
    vocab = model.config.vocab_size
    B = ids.shape[0]
    for t in prefix_lens:
        if t >= ids.shape[1]:
            continue
        top = CS.topk_candidates(model, ids, position=t, k_top=n_top)
        rand = CS.random_candidates(vocab, B, n_rand, seed=12345 + t).to(ids.device)
        cand = torch.cat([top, rand], dim=1)
        cloud = CS.candidate_cloud(model, ids, position=t, candidate_ids=cand,
                                   k_split=k, chunk_size=256)
        for b in range(cloud.shape[0]):
            U, S, r = CS.local_subspace(cloud[b], energy=energy, max_rank=max_rank)
            if r > 0:
                out.append((U.cpu(), float(S[0]), r, t))
    return out


def aggregate_subspace(local_list, max_rank=64, energy=0.9):
    if not local_list: return torch.zeros(0, 0), 0
    Us = [U for (U, _, _, _) in local_list]
    stack = torch.cat(Us, dim=1)
    U_left, S, _ = torch.linalg.svd(stack.float(), full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum().clamp(min=1e-30)
    r = int((cum >= energy).nonzero()[0].item()) + 1 if (cum >= energy).any() else len(S)
    r = max(1, min(r, max_rank, len(S)))
    return U_left[:, :r].contiguous(), r


def make_probe_directions(clean_a, n_dirs=64, seed=7):
    H = clean_a.shape[-1]
    X = clean_a.reshape(-1, H).float()
    X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    n_pca = min(n_dirs // 2, Vh.shape[0])
    pca = Vh[:n_pca]
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(n_dirs - n_pca, H, generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)
    return torch.cat([pca, rand], dim=0)


@torch.no_grad()
def run_alignment(model_id, k, n_prompts=32, prefix_lens=(2, 5, 10, 20),
                  n_probe=64, device="cuda:0", dtype=torch.float32):
    print(f"\n--- alignment {model_id}  k={k} ---", flush=True)
    safe = model_id.replace("/", "_")
    cache_p = util.ART / "activations" / safe / "sst2" / f"split_{k}" / "cache.pt"
    if not cache_p.exists():
        print(f"  skip: missing activation cache {cache_p}", flush=True)
        return None
    cache = torch.load(cache_p, weights_only=False)
    train = cache["train"]
    ids = train["ids"][:n_prompts].to(device)
    mask = train["mask"][:n_prompts].to(device)
    ans_pos = train["ans_pos"][:n_prompts].to(device)
    clean = train["clean_a"][:n_prompts]
    meta = util.ART / "activations" / safe / "sst2" / f"split_{k}" / "metadata.json"
    import json
    metadata = json.loads(meta.read_text())
    label_token_ids = metadata["label_token_ids"]

    model, _ = M.load_model(model_id, dtype=dtype, device=device)
    L = M.n_layers(model)
    if k > L:
        del model; torch.cuda.empty_cache(); return None

    local = per_prompt_local_S(model, ids, prefix_lens, k=k,
                               n_top=80, n_rand=20, energy=0.9, max_rank=24,
                               attention_mask=mask)
    U_S, r_S = aggregate_subspace(local, max_rank=64, energy=0.9)

    probes = make_probe_directions(clean, n_dirs=n_probe, seed=7)
    U_T, sigma_T, r_T = SG.label_logit_jacobian_subspace(
        model, ids, mask, ans_pos, label_token_ids, k=k,
        probe_dirs=probes, tau=1e-2, energy=0.9, max_rank=64,
    )

    angles = G.principal_angles(U_S, U_T)
    mass = G.mass_T_perp_of_S(U_S, U_T)
    deg = (angles * 180.0 / 3.141592653589793).tolist()

    out_dir = util.ART / "geometry" / safe / "sst2" / f"split_{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(U_S.cpu(), out_dir / "U_S.pt")
    torch.save(U_T.cpu(), out_dir / "U_T.pt")
    summary = {
        "model_id": model_id, "k": k, "n_prompts": int(ids.shape[0]),
        "r_S": int(r_S), "r_T": int(r_T),
        "mass_T_perp_of_S": float(mass),
        "principal_angles_deg": deg[:10],
        "decision": ("CONTINUE" if mass > 0.5 else
                     ("STOP" if mass < 0.1 else "MARGINAL")),
    }
    util.write_json(out_dir / "subspace_alignment.json", summary)
    print(f"  r_S={r_S} r_T={r_T} mass={mass:.3f} -> {summary['decision']}", flush=True)
    del model; torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--prefix-lens", nargs="+", type=int, default=[2, 5, 10, 20])
    ap.add_argument("--n-probe", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeding.set_seed(args.seed)

    rows = []
    dtype_by_model = {
        "gpt2": torch.float32,
        "Qwen/Qwen2.5-0.5B": torch.float32,
    }
    for mid in args.models:
        dt = dtype_by_model.get(mid, torch.float32)
        for k in args.ks:
            r = run_alignment(mid, k, n_prompts=args.n_prompts,
                              prefix_lens=tuple(args.prefix_lens),
                              n_probe=args.n_probe, dtype=dt, device=args.device)
            if r: rows.append(r)

    util.write_json(util.ART / "subspace_alignment_summary.json", {"rows": rows})
    rep = ["# Phase 2 SST-2 Subspace Alignment", "",
           "T_k built on answer-position label logits (2 classes).", "",
           "| model | k | r_S | r_T | mass_T_perp(S) | decision |",
           "|---|---:|---:|---:|---:|:---:|"]
    for r in rows:
        rep.append(f"| {r['model_id']} | {r['k']} | {r['r_S']} | {r['r_T']} "
                   f"| {r['mass_T_perp_of_S']:.3f} | **{r['decision']}** |")
    (util.ART / "reports" / "02_subspace_alignment.md").write_text("\n".join(rep))


if __name__ == "__main__":
    main()
