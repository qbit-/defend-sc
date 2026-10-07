"""Phase 1 SST-2: cache clean + clipped-clean cut activations and answer-position hidden states + labels. Sharded by (model, dataset, split_k)."""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from src import split_model as SM
from src import util
from src import seeding
from src.tasks import get_task
from src.tasks.privacy import add_privacy_arguments, configure_privacy


def activation_norm_quantile(
    norms: torch.Tensor, quantile: float
) -> float:
    """Quantile of activation norms in float32.

    ``torch.quantile`` accepts only float32 and float64. CUDA runs
    cache activations as bfloat16 or float16.

    Args:
        norms: Per-position activation norms.
        quantile: Quantile in ``[0, 1]``.

    Returns:
        The requested quantile as a Python float.
    """
    flat = norms.flatten().to(dtype=torch.float32)
    return float(torch.quantile(flat, quantile))


def per_position_clip(
    a: torch.Tensor, C: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clip each position's hidden state to an L2 norm of ``C``.

    Args:
        a: Activations with shape ``[batch, time, hidden]``.
        C: Maximum allowed L2 norm.

    Returns:
        Clipped activations and norm diagnostics.
    """
    norm = a.norm(dim=-1, keepdim=True)
    scale = (C / norm).clamp(max=1.0)
    clipped = a * scale
    diag = {
        "C": C,
        "frac_clipped": float((norm.squeeze(-1) > C).float().mean()),
        "norm_p50": float(norm.median()),
        "norm_p95": activation_norm_quantile(norm, 0.95),
        "norm_max": float(norm.max()),
    }
    return clipped, diag


def cache_dir(model_id, k, task_name):
    """Return the activation-cache directory for one split depth.

    Args:
        model_id: Hugging Face model id.
        k: Split depth.
        task_name: Benchmark name.

    Returns:
        ``artifacts/activations/<model>/<task>/split_<k>``.
    """
    return util.art_path(
        "activations", model_id, task_name, f"split_{k}",
    )


def _move_encoded(encoded: dict) -> dict:
    """Copy tensor fields to CPU and leave the other fields unchanged.

    Args:
        encoded: Batch returned by a task ``encode`` method.

    Returns:
        A new dict safe to store in the cache.
    """
    moved = {}
    for key, value in encoded.items():
        if torch.is_tensor(value):
            moved[key] = value.detach().cpu()
        else:
            moved[key] = value
    return moved


def _capture_clean(model, ids, mask, k, batch_size):
    """Capture cut activations for a right-padded prompt batch.

    Args:
        model: Causal language model.
        ids: Token ids.
        mask: Attention mask.
        k: Split depth.
        batch_size: Rows per forward.

    Returns:
        Cut activations on CPU.
    """
    chunks = []
    for start in range(0, ids.shape[0], batch_size):
        stop = min(start + batch_size, ids.shape[0])
        chunks.append(SM.capture_a_k(
            model, ids[start:stop], k=k, attention_mask=mask[start:stop],
        ).cpu())
    return torch.cat(chunks, dim=0)


def _pack_generation(encoded: dict, clipped: torch.Tensor) -> dict:
    """Return the generation-cache fields for one split.

    Args:
        encoded: CPU encodings plus ``clean_a``.
        clipped: Clipped prompt activations.

    Returns:
        Cache dict consumed by geometry, the suppressor, and eval.
    """
    return {
        "ids": encoded["input_ids"],
        "mask": encoded["attention_mask"],
        "ans_pos": encoded["answer_pos"],
        "clean_a": encoded["clean_a"],
        "clipped_a": clipped,
        "privacy_mask": encoded["privacy_mask"],
        "targets": encoded["targets"],
        "gold_ids": encoded["gold_ids"],
        "gold_mask": encoded["gold_mask"],
    }


@torch.no_grad()
def _capture_encoded(
    model, tokenizer, task, examples, k, max_len, device, batch_size,
):
    """Encode examples and capture their clean cut activations.

    Args:
        model: Causal language model.
        tokenizer: Model tokenizer.
        task: Generation task.
        examples: Split rows.
        k: Split depth.
        max_len: Maximum prompt length.
        device: Torch device.
        batch_size: Rows per forward.

    Returns:
        CPU encodings plus ``clean_a``.
    """
    encoded = _move_encoded(task.encode(tokenizer, examples, max_len, device))
    encoded["clean_a"] = _capture_clean(
        model,
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        k,
        batch_size,
    )
    return encoded


def _save_generation(task, model_id, k, max_len, hidden, n_layers, n_train,
                     n_test, clip_quantile, train, test, threshold,
                     diag_train, diag_test, positions) -> dict:
    """Write the generation cache and return its metadata.

    Args:
        task: Generation task.
        model_id: Hugging Face model id.
        k: Split depth.
        max_len: Prompt length used for encoding.
        hidden: Model hidden size.
        n_layers: Decoder depth.
        n_train: Requested train rows.
        n_test: Requested test rows.
        clip_quantile: Clip quantile.
        train: Captured train encodings.
        test: Captured test encodings.
        threshold: Clip threshold.
        diag_train: Train clip diagnostics.
        diag_test: Test clip diagnostics.
        positions: Privacy-span token positions.

    Returns:
        Metadata dict.
    """
    directory = cache_dir(model_id, k, task.name)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({
        "task": task.name,
        "kind": task.kind,
        "train": _pack_generation(train, train["clipped_a"]),
        "test": _pack_generation(test, test["clipped_a"]),
    }, directory / "cache.pt")
    diag = {
        "model_id": model_id, "k": k, "H": hidden, "n_layers": n_layers,
        "task": task.name, "kind": task.kind,
        "n_train": n_train, "n_test": n_test, "max_len": max_len,
        "clip_quantile": clip_quantile, "C": threshold,
        "diag_train": diag_train, "diag_test": diag_test,
        "privacy_positions": positions,
        "utility_metrics": list(task.utility_metrics),
    }
    util.write_json(directory / "metadata.json", diag)
    return diag


def collect_generation(
    task, model_id, k, n_train, n_test, max_len, dtype, device,
    clip_quantile, batch_size, seed,
):
    """Cache prompt activations for a generation benchmark.

    Args:
        task: Generation task object.
        model_id: Hugging Face model id.
        k: Split depth.
        n_train: Calibration rows.
        n_test: Eval rows.
        max_len: Maximum prompt length.
        dtype: Model dtype.
        device: Torch device.
        clip_quantile: Quantile used for the clip threshold.
        batch_size: Rows per forward.
        seed: Split subsample seed.

    Returns:
        Metadata dict, or ``None`` when ``k`` is past the last layer.
    """
    print(
        f"\n--- {task.name} {model_id}  k={k}  "
        f"N_train={n_train} N_test={n_test} ---",
        flush=True,
    )
    model, tokenizer = M.load_model(model_id, dtype=dtype, device=device)
    n_layers = M.n_layers(model)
    hidden = M.hidden_size(model)
    if k > n_layers:
        print(f"  skip k={k} > L={n_layers}")
        del model
        torch.cuda.empty_cache()
        return None
    train_ex = task.load_split("train", n_train, seed)
    test_ex = task.load_split("test", n_test, seed)
    train = _capture_encoded(
        model, tokenizer, task, train_ex, k, max_len, device, batch_size,
    )
    test = _capture_encoded(
        model, tokenizer, task, test_ex, k, max_len, device, batch_size,
    )
    threshold = activation_norm_quantile(
        train["clean_a"].norm(dim=-1), clip_quantile,
    )
    train["clipped_a"], diag_train = per_position_clip(
        train["clean_a"], threshold,
    )
    test["clipped_a"], diag_test = per_position_clip(
        test["clean_a"], threshold,
    )
    positions = task.resolve_privacy_positions({
        "input_ids": train["input_ids"],
        "attention_mask": train["attention_mask"],
        "privacy_mask": train["privacy_mask"],
    }, tokenizer)
    diag = _save_generation(
        task, model_id, k, max_len, hidden, n_layers, n_train, n_test,
        clip_quantile, train, test, threshold, diag_train, diag_test,
        positions,
    )
    print(
        f"  H={hidden}  C={threshold:.3f}  "
        f"privacy_positions={positions}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return diag


@torch.no_grad()
def collect_one(model_id, k, n_train=512, n_test=256, max_len=64,
                dtype=torch.float32, device="cuda:0", clip_quantile=0.95,
                batch_size=32, task_name="sst2", seed=0,
                attack_positions: list[int] | None = None,
                offset_pattern: str = ""):
    task = get_task(task_name)
    configure_privacy(task, attack_positions, offset_pattern)
    if task.kind == "generation":
        return collect_generation(
            task, model_id, k, n_train, n_test, max_len, dtype, device,
            clip_quantile, batch_size, seed,
        )
    print(f"\n--- {model_id}  k={k}  N_train={n_train} N_test={n_test} ---", flush=True)
    model, tok = M.load_model(model_id, dtype=dtype, device=device)
    L = M.n_layers(model); H = M.hidden_size(model)
    if k > L:
        print(f"  skip k={k} > L={L}"); del model; torch.cuda.empty_cache(); return None

    label_token_ids = task.label_token_ids(tok)
    label_words = task.label_words(tok)
    label_ids = torch.tensor(label_token_ids, device=device)

    train_ex = task.load_split("train", n_train, seed)
    test_ex = task.load_split("test", n_test, seed)

    def cache_split(examples, split_name):
        del split_name
        encoded = task.encode(tok, examples, max_len=max_len, device=device)
        ids, mask, ans_pos = (
            encoded["input_ids"], encoded["attention_mask"], encoded["answer_pos"],
        )
        labels = encoded["labels"]
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
            feat_chunks.append(task.gather_answer_hidden(server_clean_b.to(device), ans_b).cpu())
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
    C = activation_norm_quantile(norms, clip_quantile)
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
            feat_chunks.append(task.gather_answer_hidden(server_b.to(device), ans_b).cpu())
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
    d = cache_dir(model_id, k, task.name)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task.name,
        "kind": task.kind,
        "train": {
            "ids": train["ids"], "mask": train["mask"], "ans_pos": train["ans_pos"],
            "labels": train["labels"],
            "clean_a": train["clean_a"], "clipped_a": clipped_a_train,
            "server_clean": train["server_clean"], "server_clip": server_clip_tr,
            "feat_clean": train["feat_clean"], "feat_clip": feat_clip_tr,
            "label_logits_clean": train["label_logits_clean"],
            "label_logits_clip": ll_clip_tr,
            "privacy_mask": train["mask"],
        },
        "test": {
            "ids": test["ids"], "mask": test["mask"], "ans_pos": test["ans_pos"],
            "labels": test["labels"],
            "clean_a": test["clean_a"], "clipped_a": clipped_a_test,
            "server_clean": test["server_clean"], "server_clip": server_clip_te,
            "feat_clean": test["feat_clean"], "feat_clip": feat_clip_te,
            "label_logits_clean": test["label_logits_clean"],
            "label_logits_clip": ll_clip_te,
            "privacy_mask": test["mask"],
        },
    }
    torch.save(payload, d / "cache.pt")
    # Save LM-head label projection (task_weights) — used in Phase 5 fitter
    lm_head_w = M.lm_head(model).weight.detach().cpu()             # [V, H]
    task_weights = lm_head_w[label_ids.cpu()]                      # [2, H]
    torch.save(task_weights, d / "task_weights.pt")

    # Report
    positions = task.resolve_privacy_positions(
        {
            "input_ids": train["ids"],
            "attention_mask": train["mask"],
            "privacy_mask": train["mask"],
        },
        tok,
    )
    diag = {
        "model_id": model_id, "k": k, "H": H, "n_layers": L,
        "task": task.name, "kind": task.kind,
        "n_train": n_train, "n_test": n_test,
        "clip_quantile": clip_quantile, "C": C,
        "diag_train": diag_train, "diag_test": diag_test,
        "verbalizer": label_words,
        "label_token_ids": label_token_ids,
        "privacy_positions": positions,
        "utility_metrics": list(task.utility_metrics),
    }
    util.write_json(d / "metadata.json", diag)
    print(f"  H={H}  C={C:.3f}  frac_clipped(train)={diag_train['frac_clipped']:.3f}  "
          f"verb_ids={label_token_ids}", flush=True)
    del model; torch.cuda.empty_cache()
    return diag


def _write_calibration_reports(task_name: str, rows: list[dict]) -> None:
    """Write one calibration report per model.

    Args:
        task_name: Benchmark name.
        rows: Metadata rows returned by ``collect_one``.
    """
    models = []
    for row in rows:
        if row["model_id"] not in models:
            models.append(row["model_id"])
    for model_id in models:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        report = [
            f"# Phase 1 {task_name} Calibration",
            "",
            "| model | k | C | frac_clipped(tr) | norm_p50 | norm_p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in model_rows:
            report.append(
                f"| {row['model_id']} | {row['k']} | {row['C']:.3f} "
                f"| {row['diag_train']['frac_clipped']:.3f} "
                f"| {row['diag_train']['norm_p50']:.2f} "
                f"| {row['diag_train']['norm_p95']:.2f} |"
            )
        directory = util.art_path("reports", model_id, task_name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "01_collect_calibration.md").write_text(
            "\n".join(report),
        )
        util.write_json(
            directory / "calibration_summary.json", {"rows": model_rows},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--clip-quantile", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=M.DEFAULT_BATCH)
    ap.add_argument("--seed", type=int, default=0)
    add_privacy_arguments(ap)
    args = ap.parse_args()
    seeding.set_seed(args.seed)
    task = get_task(args.task)
    max_len = task.max_len if args.max_len is None else args.max_len

    rows = []
    for mid in args.models:
        dt = M.default_dtype(mid, args.device)
        for k in args.ks:
            r = collect_one(mid, k, n_train=args.n_train, n_test=args.n_test,
                            max_len=max_len, dtype=dt, device=args.device,
                            clip_quantile=args.clip_quantile,
                            batch_size=args.batch_size,
                            task_name=args.task, seed=args.seed,
                            attack_positions=args.positions,
                            offset_pattern=args.offset_pattern)
            if r: rows.append(r)
    _write_calibration_reports(args.task, rows)


if __name__ == "__main__":
    main()
