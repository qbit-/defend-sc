"""Measure inference slowdown from low-rank noise and suppression."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

try:
    import pyarrow as pa

    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

from src import sst2_data as D
from src import util


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--split-k", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--measure-iters", type=int, default=5)
    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches for smoke tests; 0 uses all batches.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=util.ART / "inference_slowdown",
    )
    ap.add_argument("--prefix", default="qwen_inference_slowdown")
    return ap.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Resolve a dtype name into a torch dtype."""
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def model_label(model_id: str) -> str:
    """Return a compact label for a Qwen model id."""
    return model_id.rsplit("/", maxsplit=1)[-1].replace("Qwen2.5-", "")


def load_batches(
    tokenizer: AutoTokenizer,
    n_prompts: int,
    max_len: int,
    batch_size: int,
    device: str,
    seed: int,
    max_batches: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load SST-2 validation prompts and return device batches."""
    examples = D.load_sst2(split="validation", n=n_prompts, seed=seed)
    ids, mask, _ = D.encode_prompts(
        tokenizer,
        examples,
        max_len=max_len,
        device=device,
    )
    batches = []
    for start in range(0, ids.shape[0], batch_size):
        if max_batches and len(batches) >= max_batches:
            break
        end = min(start + batch_size, ids.shape[0])
        batches.append((ids[start:end], mask[start:end]))
    return batches


def qwen_context(
    qwen_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Create Qwen hidden states, positions, rotary embeddings, and masks."""
    hidden_states = qwen_model.embed_tokens(input_ids)
    cache_position = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
    )
    position_ids = cache_position.unsqueeze(0)
    mask_kwargs = {
        "config": qwen_model.config,
        "input_embeds": hidden_states,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_masks = {
        "full_attention": modeling_qwen2.create_causal_mask(**mask_kwargs),
    }
    if getattr(qwen_model, "has_sliding_layers", False):
        causal_masks["sliding_attention"] = (
            modeling_qwen2.create_sliding_window_causal_mask(**mask_kwargs)
        )
    position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)
    return hidden_states, position_ids, cache_position, {
        "causal_masks": causal_masks,
        "position_embeddings": position_embeddings,
    }


def run_qwen_layers(
    layers: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    causal_masks: dict,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run a sequence of Qwen decoder layers."""
    for layer in layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_masks[layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
    return hidden_states


def qwen_head(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    split_k: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, dict]]:
    """Run Qwen embeddings and layers before the split."""
    qwen_model = model.model
    hidden, position_ids, cache_position, extra = qwen_context(
        qwen_model,
        input_ids,
        attention_mask,
    )
    hidden = run_qwen_layers(
        qwen_model.layers[:split_k],
        hidden,
        position_ids,
        cache_position,
        extra["causal_masks"],
        extra["position_embeddings"],
    )
    return hidden, (position_ids, cache_position, extra)


def qwen_tail(
    model: Any,
    split_hidden: torch.Tensor,
    split_k: int,
    context: tuple[torch.Tensor, torch.Tensor, dict],
) -> torch.Tensor:
    """Run Qwen layers after the split, final norm, and LM head."""
    qwen_model = model.model
    position_ids, cache_position, extra = context
    hidden = run_qwen_layers(
        qwen_model.layers[split_k:],
        split_hidden,
        position_ids,
        cache_position,
        extra["causal_masks"],
        extra["position_embeddings"],
    )
    hidden = qwen_model.norm(hidden)
    return model.lm_head(hidden)


def make_placeholder_state(
    hidden_size: int,
    rank: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create random low-rank noise and suppressor placeholders."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn(hidden_size, rank, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    u_eta = q.to(device=device, dtype=dtype)
    lam = torch.ones(rank, device=device, dtype=dtype)
    gamma = torch.full((rank,), 0.5, device=device, dtype=dtype)
    mean = torch.zeros(hidden_size, device=device, dtype=dtype)
    return {"U_eta": u_eta, "lam": lam, "gamma": gamma, "mean": mean}


def add_lowrank_noise(
    hidden: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate and add placeholder low-rank Gaussian noise."""
    rank = state["U_eta"].shape[1]
    z = torch.randn(
        *hidden.shape[:-1],
        rank,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    eta = (z * state["lam"].sqrt()) @ state["U_eta"].T
    return hidden + eta


def apply_suppressor(
    hidden: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the placeholder private low-rank suppressor."""
    centered = hidden - state["mean"]
    coeff = centered @ state["U_eta"]
    return hidden - (coeff * state["gamma"]) @ state["U_eta"].T


def sync_if_needed(device: str) -> None:
    """Synchronize CUDA timing when needed."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(
    fn: Callable[[], None],
    device: str,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    """Return mean elapsed seconds per measured iteration."""
    for _ in range(warmup_iters):
        fn()
    sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(measure_iters):
        fn()
    sync_if_needed(device)
    return (time.perf_counter() - start) / max(1, measure_iters)


def load_model(
    model_id: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Hugging Face causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, tokenizer


def benchmark_model(
    model_id: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict:
    """Benchmark one model and return a result row."""
    model, tokenizer = load_model(model_id, dtype, args.device)
    qwen_model = model.model
    n_layers = len(qwen_model.layers)
    if args.split_k > n_layers:
        raise ValueError(f"split_k={args.split_k} exceeds n_layers={n_layers}")
    batches = load_batches(
        tokenizer,
        args.n_prompts,
        args.max_len,
        args.batch_size,
        args.device,
        args.seed,
        args.max_batches,
    )
    hidden_size = int(model.config.hidden_size)
    state = make_placeholder_state(
        hidden_size,
        args.rank,
        args.device,
        dtype,
        args.seed,
    )

    def baseline_run() -> None:
        for ids, mask in batches:
            model(input_ids=ids, attention_mask=mask)

    split_inputs = []
    with torch.inference_mode():
        for ids, mask in batches:
            hidden, context = qwen_head(model, ids, mask, args.split_k)
            split_inputs.append((hidden.detach(), context))

    def head_run() -> None:
        for ids, mask in batches:
            qwen_head(model, ids, mask, args.split_k)

    def noise_run() -> None:
        for hidden, _ in split_inputs:
            add_lowrank_noise(hidden, state)

    noisy_inputs = []
    with torch.inference_mode():
        for hidden, context in split_inputs:
            noisy_inputs.append((add_lowrank_noise(hidden, state), context))

    def suppressor_run() -> None:
        for noisy, _ in noisy_inputs:
            apply_suppressor(noisy, state)

    denoised_inputs = []
    with torch.inference_mode():
        for noisy, context in noisy_inputs:
            denoised_inputs.append((apply_suppressor(noisy, state), context))

    def tail_run() -> None:
        for hidden, context in denoised_inputs:
            qwen_tail(model, hidden, args.split_k, context)

    with torch.inference_mode():
        baseline_s = time_callable(
            baseline_run,
            args.device,
            args.warmup_iters,
            args.measure_iters,
        )
        head_s = time_callable(
            head_run,
            args.device,
            args.warmup_iters,
            args.measure_iters,
        )
        noise_s = time_callable(
            noise_run,
            args.device,
            args.warmup_iters,
            args.measure_iters,
        )
        suppressor_s = time_callable(
            suppressor_run,
            args.device,
            args.warmup_iters,
            args.measure_iters,
        )
        tail_s = time_callable(
            tail_run,
            args.device,
            args.warmup_iters,
            args.measure_iters,
        )

    split_s = head_s + noise_s + suppressor_s + tail_s
    n_examples = sum(int(ids.shape[0]) for ids, _ in batches)
    return {
        "model_id": model_id,
        "model_label": model_label(model_id),
        "status": "ok",
        "error": "",
        "dtype": str(dtype).replace("torch.", ""),
        "device": args.device,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "split_k": args.split_k,
        "rank": args.rank,
        "n_examples": n_examples,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "baseline_full_s": baseline_s,
        "head_to_split_s": head_s,
        "noise_generation_s": noise_s,
        "suppressor_s": suppressor_s,
        "tail_from_split_s": tail_s,
        "split_total_s": split_s,
        "slowdown": split_s / baseline_s if baseline_s > 0 else float("nan"),
    }


def failed_row(model_id: str, args: argparse.Namespace, exc: Exception) -> dict:
    """Create a result row for a failed benchmark."""
    return {
        "model_id": model_id,
        "model_label": model_label(model_id),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "dtype": args.dtype,
        "device": args.device,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write benchmark rows to CSV."""
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    """Write benchmark metadata and rows to JSON."""
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2))


def plot_rows(rows: list[dict], out_dir: Path, prefix: str) -> list[Path]:
    """Plot slowdown ratios and component timings."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    plot_dir = util.ART / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["model_label"] for row in ok_rows]
    slowdowns = [float(row["slowdown"]) for row in ok_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, slowdowns, alpha=0.8)
    ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_ylabel("split + noise + suppressor / full inference")
    ax.set_xlabel("Qwen2.5 model size")
    ax.set_title("Inference Slowdown From Low-Rank Noise Suppression")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(slowdowns):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom")
    paths = [
        out_dir / f"{prefix}_bar.png",
        plot_dir / f"{prefix}_bar.png",
    ]
    for path in paths:
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        print(f"wrote {path}", flush=True)
    plt.close(fig)

    noise_times = [float(row["noise_generation_s"]) for row in ok_rows]
    suppressor_times = [float(row["suppressor_s"]) for row in ok_rows]
    x_pos = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [x - width / 2 for x in x_pos],
        noise_times,
        width,
        label="noise addition",
        alpha=0.8,
    )
    ax.bar(
        [x + width / 2 for x in x_pos],
        suppressor_times,
        width,
        label="suppression",
        alpha=0.8,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds per benchmark iteration")
    ax.set_xlabel("Qwen2.5 model size")
    ax.set_title("Absolute Noise Addition and Suppression Time")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    timing_paths = [
        out_dir / f"{prefix}_noise_suppressor_times.png",
        plot_dir / f"{prefix}_noise_suppressor_times.png",
    ]
    for path in timing_paths:
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        print(f"wrote {path}", flush=True)
    plt.close(fig)
    paths.extend(timing_paths)
    return paths


def cleanup_model() -> None:
    """Release Python and CUDA memory between model benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run the inference slowdown benchmark."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype, args.device)
    torch.manual_seed(args.seed)
    rows = []
    for model_id in args.models:
        print(f"\n--- benchmarking {model_id} ---", flush=True)
        try:
            rows.append(benchmark_model(model_id, args, dtype))
        except Exception as exc:
            print(f"FAILED {model_id}: {exc}", flush=True)
            rows.append(failed_row(model_id, args, exc))
        finally:
            cleanup_model()

    csv_path = args.out_dir / f"{args.prefix}.csv"
    json_path = args.out_dir / f"{args.prefix}.json"
    write_csv(csv_path, rows)
    write_json(json_path, args, rows)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    plot_rows(rows, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
