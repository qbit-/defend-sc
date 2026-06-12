"""Phase 1 SST-2: cache clean + clipped-clean cut activations and answer-position hidden states + labels. Sharded by (model, dataset, split_k)."""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from src import split_model as SM
from src import sst2_data as D
from src import metrics as MET
from src import util
from src import seeding


def per_position_clip(a: torch.Tensor, C: float):
    norm = a.norm(dim=-1, keepdim=True)
    scale = (C / norm).clamp(max=1.0)
    clipped = a * scale
    diag = {
        "C": C,
        "frac_clipped": float((norm.squeeze(-1) > C).float().mean()),
        "norm_p50": float(norm.median()),
        "norm_p95": float(torch.quantile(norm.flatten(), 0.95)),
        "norm_max": float(norm.max()),
    }
    return clipped, diag


def cache_dir(model_id, k):
    safe = model_id.replace("/", "_")
    return util.ART / "activations" / safe / "sst2" / f"split_{k}"


@torch.no_grad()
def collect_one(model_id, k, n_train=512, n_test=256, max_len=64,
                dtype=torch.float32, device="cuda:0", clip_quantile=0.95,
                batch_size=32):
    print(f"\n--- {model_id}  k={k}  N_train={n_train} N_test={n_test} ---", flush=True)
    model, tok = M.load_model(model_id, dtype=dtype, device=device)
    L = M.n_layers(model); H = M.hidden_size(model)
    if k > L:
        print(f"  skip k={k} > L={L}"); del model; torch.cuda.empty_cache(); return None

    verb = D.build_verbalizer(tok)
    label_ids = torch.tensor(verb.label_token_ids, device=device)

    train_ex = D.load_sst2(split="train", n=n_train, seed=0)
    test_ex = D.load_sst2(split="validation", n=n_test, seed=0)

    def cache_split(examples, split_name):
        ids, mask, ans_pos = D.encode_prompts(tok, examples, max_len=max_len, device=device)
        labels = D.labels_tensor(examples, device=device)
        clean_chunks, server_chunks, feat_chunks, logit_chunks = [], [], [], []
        for start in range(0, ids.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, ids.shape[0]))
            ids_b = ids[sl]
            mask_b = mask[sl]
            ans_b = ans_pos[sl]
            clean_a_b = SM.capture_a_k(model, ids_b, k=k, attention_mask=mask_b).cpu()
            out_clean = SM.split_run(model, ids_b, k=k, attention_mask=mask_b)
            server_clean_b = out_clean["server_out"].cpu()
            logits_clean_full = out_clean["logits"]                     # [B, T, V]
            pos = ans_b.view(-1, 1, 1).expand(-1, 1, logits_clean_full.shape[-1])
            last_clean = logits_clean_full.gather(1, pos).squeeze(1)
            clean_chunks.append(clean_a_b)
            server_chunks.append(server_clean_b)
            feat_chunks.append(D.gather_answer_position(server_clean_b.to(device), ans_b).cpu())
            logit_chunks.append(last_clean.index_select(-1, label_ids).cpu())
            del out_clean, logits_clean_full, last_clean
            torch.cuda.empty_cache()
        clean_a = torch.cat(clean_chunks, dim=0)
        server_clean = torch.cat(server_chunks, dim=0)
        feat_clean = torch.cat(feat_chunks, dim=0)
        label_logits_clean = torch.cat(logit_chunks, dim=0)
        return {
            "ids": ids.cpu(), "mask": mask.cpu(), "ans_pos": ans_pos.cpu(),
            "labels": labels.cpu(),
            "clean_a": clean_a, "server_clean": server_clean,
            "feat_clean": feat_clean, "label_logits_clean": label_logits_clean,
        }

    train = cache_split(train_ex, "train")
    test = cache_split(test_ex, "test")

    # Choose clip C from train clean norms
    norms = train["clean_a"].norm(dim=-1)
    C = float(torch.quantile(norms.flatten(), clip_quantile))
    clipped_a_train, diag_train = per_position_clip(train["clean_a"], C)
    clipped_a_test, diag_test = per_position_clip(test["clean_a"], C)

    # Replay with clipped a_k for both splits
    def replay(clipped_a, ids, mask, ans_pos):
        server_chunks, feat_chunks, logit_chunks = [], [], []
        for start in range(0, ids.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, ids.shape[0]))
            ids_b = ids[sl].to(device)
            mask_b = mask[sl].to(device)
            ans_b = ans_pos[sl].to(device)
            clipped_b = clipped_a[sl].to(device, dtype)
            out = SM.split_run(model, ids_b, k=k, hidden_override=clipped_b,
                               attention_mask=mask_b)
            server_b = out["server_out"].cpu()
            pos = ans_b.view(-1, 1, 1).expand(-1, 1, out["logits"].shape[-1])
            last_clip = out["logits"].gather(1, pos).squeeze(1)
            server_chunks.append(server_b)
            feat_chunks.append(D.gather_answer_position(server_b.to(device), ans_b).cpu())
            logit_chunks.append(last_clip.index_select(-1, label_ids).cpu())
            del out, last_clip
            torch.cuda.empty_cache()
        server_clip = torch.cat(server_chunks, dim=0)
        feat_clip = torch.cat(feat_chunks, dim=0)
        label_logits_clip = torch.cat(logit_chunks, dim=0)
        return server_clip, feat_clip, label_logits_clip

    server_clip_tr, feat_clip_tr, ll_clip_tr = replay(
        clipped_a_train, train["ids"], train["mask"], train["ans_pos"])
    server_clip_te, feat_clip_te, ll_clip_te = replay(
        clipped_a_test, test["ids"], test["mask"], test["ans_pos"])

    # Save
    d = cache_dir(model_id, k)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": {
            "ids": train["ids"], "mask": train["mask"], "ans_pos": train["ans_pos"],
            "labels": train["labels"],
            "clean_a": train["clean_a"], "clipped_a": clipped_a_train,
            "server_clean": train["server_clean"], "server_clip": server_clip_tr,
            "feat_clean": train["feat_clean"], "feat_clip": feat_clip_tr,
            "label_logits_clean": train["label_logits_clean"],
            "label_logits_clip": ll_clip_tr,
        },
        "test": {
            "ids": test["ids"], "mask": test["mask"], "ans_pos": test["ans_pos"],
            "labels": test["labels"],
            "clean_a": test["clean_a"], "clipped_a": clipped_a_test,
            "server_clean": test["server_clean"], "server_clip": server_clip_te,
            "feat_clean": test["feat_clean"], "feat_clip": feat_clip_te,
            "label_logits_clean": test["label_logits_clean"],
            "label_logits_clip": ll_clip_te,
        },
    }
    torch.save(payload, d / "cache.pt")
    # Save LM-head label projection (task_weights) — used in Phase 5 fitter
    lm_head_w = M.lm_head(model).weight.detach().cpu()             # [V, H]
    task_weights = lm_head_w[label_ids.cpu()]                      # [2, H]
    torch.save(task_weights, d / "task_weights.pt")

    # Report
    diag = {
        "model_id": model_id, "k": k, "H": H, "n_layers": L,
        "n_train": n_train, "n_test": n_test,
        "clip_quantile": clip_quantile, "C": C,
        "diag_train": diag_train, "diag_test": diag_test,
        "verbalizer": verb.label_words,
        "label_token_ids": verb.label_token_ids,
    }
    util.write_json(d / "metadata.json", diag)
    print(f"  H={H}  C={C:.3f}  frac_clipped(train)={diag_train['frac_clipped']:.3f}  "
          f"verb_ids={verb.label_token_ids}", flush=True)
    del model; torch.cuda.empty_cache()
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--clip-quantile", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=32)
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
            r = collect_one(mid, k, n_train=args.n_train, n_test=args.n_test,
                            max_len=args.max_len, dtype=dt, device=args.device,
                            clip_quantile=args.clip_quantile,
                            batch_size=args.batch_size)
            if r: rows.append(r)
    util.write_json(util.ART / "calibration_summary.json", {"rows": rows})
    rep = ["# Phase 1 SST-2 Calibration", "", "| model | k | C | frac_clipped(tr) | norm_p50 | norm_p95 |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        rep.append(f"| {r['model_id']} | {r['k']} | {r['C']:.3f} "
                   f"| {r['diag_train']['frac_clipped']:.3f} "
                   f"| {r['diag_train']['norm_p50']:.2f} "
                   f"| {r['diag_train']['norm_p95']:.2f} |")
    (util.ART / "reports" / "01_collect_calibration.md").write_text("\n".join(rep))


if __name__ == "__main__":
    main()
