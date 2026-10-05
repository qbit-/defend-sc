"""Measure inference slowdown from low-rank noise and suppression.

Head and tail times come from one model forward. A pre-hook at the
split records when the cut activation is reached. The same path works
for Qwen2.5 and Qwen3.5.
"""
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

from src import models as M
from src import util
from src.tasks import get_task


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3.5-4B",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
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
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=M.DEFAULT_BATCH)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--measure-iters", type=int, default=5)
    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches for smoke tests; 0 uses all batches.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default="inference_slowdown")
    return ap.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Resolve a dtype name into a torch dtype.

    Args:
        name: ``auto``, ``float32``, ``float16``, or ``bfloat16``.
        device: Torch device string.

    Returns:
        Torch dtype. ``auto`` follows ``default_dtype``.
    """
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "auto":
        return M.default_dtype("", device)
    return torch.float32


def model_label(model_id: str) -> str:
    """Return a short label for a model id.

    Args:
        model_id: Hugging Face model id.

    Returns:
        Repository name without the organization.
    """
    return model_id.rsplit("/", maxsplit=1)[-1]


def load_batches(
    tokenizer: Any,
    n_prompts: int,
    max_len: int,
    batch_size: int,
    device: str,
    seed: int,
    max_batches: int,
    task_name: str = "sst2",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load benchmark prompts and return device batches.

    Args:
        tokenizer: Tokenizer used to encode prompts.
        n_prompts: Number of eval prompts.
        max_len: Maximum token length.
        batch_size: Prompts per forward.
        device: Torch device string.
        seed: Shuffle seed.
        max_batches: Cap on batches. ``0`` keeps all batches.
        task_name: Benchmark name.

    Returns:
        List of ``(input_ids, attention_mask)`` batches.
    """
    task = get_task(task_name)
    examples = task.load_split("test", n_prompts, seed)
    encoded = task.encode(tokenizer, examples, max_len, device)
    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]
    batches = []
    for start in range(0, ids.shape[0], batch_size):
        if max_batches and len(batches) >= max_batches:
            break
        end = min(start + batch_size, ids.shape[0])
        batches.append((ids[start:end], mask[start:end]))
    return batches


def cut_module(model: torch.nn.Module, split_k: int) -> torch.nn.Module:
    """Return the module whose input is the cut activation.

    Args:
        model: Loaded model.
        split_k: Split depth, from 0 through the layer count.

    Returns:
        Decoder block ``split_k``, or the final norm at the last cut.
    """
    n_layers = M.n_layers(model)
    if not 0 <= split_k <= n_layers:
        raise ValueError(f"split_k={split_k} is outside 0..{n_layers}")
    if split_k < n_layers:
        return M.get_blocks(model)[split_k]
    return M.final_norm(model)


def sync_if_needed(device: str) -> None:
    """Synchronize CUDA timing when the device is CUDA.

    Args:
        device: Torch device string.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _hidden_from_call(args: tuple, kwargs: dict) -> torch.Tensor:
    """Return the hidden state passed into a hooked module.

    Args:
        args: Positional arguments from the pre-hook.
        kwargs: Keyword arguments from the pre-hook.

    Returns:
        Hidden-state tensor.
    """
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return kwargs["hidden_states"]


def time_one_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    split_k: int,
    device: str,
) -> tuple[float, float, torch.Tensor]:
    """Time one forward up to the split and after it.

    Args:
        model: Eval-mode model.
        input_ids: Token ids for one batch.
        attention_mask: Attention mask for that batch.
        split_k: Decoder depth where noise is injected.
        device: Torch device string.

    Returns:
        Head seconds, tail seconds, and the cut activation.
    """
    captured: dict[str, Any] = {}

    def hook(
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        sync_if_needed(device)
        captured["t"] = time.perf_counter()
        hidden = _hidden_from_call(args, kwargs)
        captured["h"] = hidden.detach().clone()

    target = cut_module(model, split_k)
    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    sync_if_needed(device)
    start = time.perf_counter()
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    sync_if_needed(device)
    end = time.perf_counter()
    if "t" not in captured or "h" not in captured:
        raise RuntimeError(f"split hook did not run at k={split_k}")
    head = float(captured["t"]) - start
    tail = end - float(captured["t"])
    return head, tail, captured["h"]


def average_split_times(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    split_k: int,
    device: str,
    warmup_iters: int,
    measure_iters: int,
) -> tuple[float, float, list[torch.Tensor]]:
    """Average head and tail time over measured passes.

    Args:
        model: Eval-mode model.
        batches: Device batches of ``(input_ids, mask)``.
        split_k: Split depth.
        device: Torch device string.
        warmup_iters: Untimed passes.
        measure_iters: Timed passes. Must be positive.

    Returns:
        Mean head seconds, mean tail seconds, and cut activations
        from the last measured pass.
    """
    saved: list[torch.Tensor] = []

    def once() -> tuple[float, float]:
        head = 0.0
        tail = 0.0
        saved.clear()
        with torch.inference_mode():
            for ids, mask in batches:
                head_s, tail_s, hidden = time_one_forward(
                    model, ids, mask, split_k, device,
                )
                head += head_s
                tail += tail_s
                saved.append(hidden)
        return head, tail

    for _ in range(warmup_iters):
        once()
    heads: list[float] = []
    tails: list[float] = []
    for _ in range(measure_iters):
        head, tail = once()
        heads.append(head)
        tails.append(tail)
    if not heads:
        raise ValueError("measure_iters must be positive")
    scale = float(len(heads))
    return sum(heads) / scale, sum(tails) / scale, list(saved)


def make_placeholder_state(
    hidden_size: int,
    rank: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create random low-rank noise and suppressor placeholders.

    Args:
        hidden_size: Activation width.
        rank: Noise rank.
        device: Torch device string.
        dtype: Tensor dtype.
        seed: CPU generator seed.

    Returns:
        ``U_eta``, ``lam``, ``gamma``, and ``mean`` tensors.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn(hidden_size, rank, generator=gen, dtype=torch.float32)
    q_factor, _ = torch.linalg.qr(basis, mode="reduced")
    u_eta = q_factor.to(device=device, dtype=dtype)
    lam = torch.ones(rank, device=device, dtype=dtype)
    gamma = torch.full((rank,), 0.5, device=device, dtype=dtype)
    mean = torch.zeros(hidden_size, device=device, dtype=dtype)
    return {"U_eta": u_eta, "lam": lam, "gamma": gamma, "mean": mean}


def add_lowrank_noise(
    hidden: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate and add placeholder low-rank Gaussian noise.

    Args:
        hidden: Cut activation.
        state: Placeholder noise tensors.

    Returns:
        Noisy activation of the same shape.
    """
    rank = state["U_eta"].shape[1]
    factors = torch.randn(
        *hidden.shape[:-1],
        rank,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    eta = (factors * state["lam"].sqrt()) @ state["U_eta"].T
    return hidden + eta


def apply_suppressor(
    hidden: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the placeholder private low-rank suppressor.

    Args:
        hidden: Noisy cut activation.
        state: Placeholder suppressor tensors.

    Returns:
        Suppressed activation of the same shape.
    """
    centered = hidden - state["mean"]
    coeff = centered @ state["U_eta"]
    return hidden - (coeff * state["gamma"]) @ state["U_eta"].T


def time_callable(
    fn: Callable[[], None],
    device: str,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    """Return mean elapsed seconds per measured iteration.

    Args:
        fn: Zero-argument timed call.
        device: Torch device string.
        warmup_iters: Untimed calls.
        measure_iters: Timed calls.

    Returns:
        Mean seconds per measured call.
    """
    for _ in range(warmup_iters):
        fn()
    sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(measure_iters):
        fn()
    sync_if_needed(device)
    return (time.perf_counter() - start) / max(1, measure_iters)


def measure_components(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    state: dict[str, torch.Tensor],
    split_k: int,
    device: str,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, float]:
    """Time full inference and the split noise protocol.

    Args:
        model: Eval-mode model.
        batches: Device batches of ``(input_ids, mask)``.
        state: Placeholder noise and suppressor tensors.
        split_k: Split depth.
        device: Torch device string.
        warmup_iters: Untimed passes.
        measure_iters: Timed passes.

    Returns:
        Mean seconds for each timed component.
    """
    def baseline_run() -> None:
        with torch.inference_mode():
            for ids, mask in batches:
                model(input_ids=ids, attention_mask=mask)

    baseline_s = time_callable(
        baseline_run, device, warmup_iters, measure_iters,
    )
    head_s, tail_s, hiddens = average_split_times(
        model, batches, split_k, device, warmup_iters, measure_iters,
    )

    def noise_run() -> None:
        with torch.inference_mode():
            for hidden in hiddens:
                add_lowrank_noise(hidden, state)

    noise_s = time_callable(
        noise_run, device, warmup_iters, measure_iters,
    )
    with torch.inference_mode():
        noisy = [add_lowrank_noise(hidden, state) for hidden in hiddens]

    def suppressor_run() -> None:
        with torch.inference_mode():
            for hidden in noisy:
                apply_suppressor(hidden, state)

    suppressor_s = time_callable(
        suppressor_run, device, warmup_iters, measure_iters,
    )
    return {
        "baseline_full_s": baseline_s,
        "head_to_split_s": head_s,
        "noise_generation_s": noise_s,
        "suppressor_s": suppressor_s,
        "tail_from_split_s": tail_s,
    }


def result_row(
    model_id: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    n_layers: int,
    hidden_size: int,
    batch_size: int,
    n_examples: int,
    times: dict[str, float],
) -> dict[str, Any]:
    """Build one successful benchmark row.

    Args:
        model_id: Hugging Face model id.
        args: Parsed CLI arguments.
        dtype: Parameter dtype.
        n_layers: Decoder depth.
        hidden_size: Activation width.
        batch_size: Prompts per forward.
        n_examples: Number of timed prompts.
        times: Component times from ``measure_components``.

    Returns:
        CSV-ready measurement row.
    """
    split_s = (
        times["head_to_split_s"]
        + times["noise_generation_s"]
        + times["suppressor_s"]
        + times["tail_from_split_s"]
    )
    baseline = times["baseline_full_s"]
    slowdown = split_s / baseline if baseline > 0 else float("nan")
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
        "batch_size": batch_size,
        "max_len": args.max_len,
        "split_total_s": split_s,
        "slowdown": slowdown,
        **times,
    }


def benchmark_model(
    model_id: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Benchmark one model and return a result row.

    Args:
        model_id: Hugging Face model id.
        args: Parsed CLI arguments.
        dtype: Parameter dtype.

    Returns:
        CSV-ready measurement row.
    """
    model, tokenizer = M.load_model(
        model_id, dtype=dtype, device=args.device,
    )
    n_layers = M.n_layers(model)
    if args.split_k > n_layers:
        raise ValueError(
            f"split_k={args.split_k} exceeds n_layers={n_layers}"
        )
    batch_size = args.batch_size
    batches = load_batches(
        tokenizer,
        args.n_prompts,
        args.max_len,
        batch_size,
        args.device,
        args.seed,
        args.max_batches,
        args.task,
    )
    if not batches:
        raise ValueError(f"no batches for {model_id}")
    hidden = M.hidden_size(model)
    state = make_placeholder_state(
        hidden, args.rank, args.device, dtype, args.seed,
    )
    times = measure_components(
        model,
        batches,
        state,
        args.split_k,
        args.device,
        args.warmup_iters,
        args.measure_iters,
    )
    n_examples = sum(int(ids.shape[0]) for ids, _ in batches)
    row = result_row(
        model_id, args, dtype, n_layers, hidden, batch_size,
        n_examples, times,
    )
    del model, tokenizer
    return row


def failed_row(
    model_id: str,
    args: argparse.Namespace,
    exc: Exception,
) -> dict[str, Any]:
    """Create a result row for a failed benchmark.

    Args:
        model_id: Hugging Face model id.
        args: Parsed CLI arguments.
        exc: Exception raised by the benchmark.

    Returns:
        CSV-ready failure row.
    """
    return {
        "model_id": model_id,
        "model_label": model_label(model_id),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "dtype": args.dtype,
        "device": args.device,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write benchmark rows to CSV.

    Args:
        path: Destination CSV path.
        rows: Result rows.
    """
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict],
) -> None:
    """Write benchmark metadata and rows to JSON.

    Args:
        path: Destination JSON path.
        args: Parsed CLI arguments.
        rows: Result rows.
    """
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2))


def _save_fig(fig: Any, paths: list[Path]) -> None:
    """Save ``fig`` to each path and close it.

    Args:
        fig: Matplotlib figure.
        paths: Destination PNG paths.
    """
    for path in paths:
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def plot_slowdown(
    rows: list[dict],
    out_dir: Path,
    prefix: str,
    plot_dir: Path,
) -> list[Path]:
    """Plot the split-to-full slowdown ratio.

    Args:
        rows: Successful benchmark rows.
        out_dir: Directory for a copy of the figure.
        prefix: Filename prefix.

    Returns:
        Written PNG paths.
    """
    labels = [row["model_label"] for row in rows]
    slowdowns = [float(row["slowdown"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, slowdowns, alpha=0.8)
    ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_ylabel("split + noise + suppressor / full inference")
    ax.set_xlabel("model")
    ax.set_title("Inference Slowdown From Low-Rank Noise Suppression")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(slowdowns):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom")
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / f"{prefix}_bar.png",
        plot_dir / f"{prefix}_bar.png",
    ]
    _save_fig(fig, paths)
    return paths


def plot_component_times(
    rows: list[dict],
    out_dir: Path,
    prefix: str,
    plot_dir: Path,
) -> list[Path]:
    """Plot noise-addition and suppression time.

    Args:
        rows: Successful benchmark rows.
        out_dir: Directory for a copy of the figure.
        prefix: Filename prefix.

    Returns:
        Written PNG paths.
    """
    labels = [row["model_label"] for row in rows]
    noise_times = [float(row["noise_generation_s"]) for row in rows]
    suppressor_times = [float(row["suppressor_s"]) for row in rows]
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
    ax.set_xlabel("model")
    ax.set_title("Absolute Noise Addition and Suppression Time")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    paths = [
        out_dir / f"{prefix}_noise_suppressor_times.png",
        plot_dir / f"{prefix}_noise_suppressor_times.png",
    ]
    _save_fig(fig, paths)
    return paths


def plot_rows(
    rows: list[dict],
    out_dir: Path,
    prefix: str,
    plot_dir: Path,
) -> list[Path]:
    """Plot slowdown ratios and component timings.

    Args:
        rows: Benchmark rows, including failures.
        out_dir: Directory for copies of the figures.
        prefix: Filename prefix.
        plot_dir: Plot directory.

    Returns:
        Written PNG paths. Empty when every row failed.
    """
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    paths = plot_slowdown(ok_rows, out_dir, prefix, plot_dir)
    paths.extend(plot_component_times(ok_rows, out_dir, prefix, plot_dir))
    return paths


def cleanup_model() -> None:
    """Release Python and CUDA memory between model benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slowdown_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Choose the CSV directory and the plot directory.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Output directory and plot directory.
    """
    if args.out_dir is not None:
        return args.out_dir, args.out_dir
    if len(args.models) == 1:
        model_id = args.models[0]
        return (
            util.art_path("inference_slowdown", model_id, args.task),
            util.art_path("plots", model_id, args.task),
        )
    return (
        util.ART / "inference_slowdown" / "comparison" / args.task,
        util.ART / "plots" / "comparison" / args.task,
    )


def main() -> None:
    """Run the inference slowdown benchmark."""
    args = parse_args()
    task = get_task(args.task)
    if args.max_len is None:
        args.max_len = task.max_len
    args.out_dir, plot_dir = _slowdown_dirs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rows = []
    for model_id in args.models:
        print(f"\n--- benchmarking {model_id} ---", flush=True)
        dtype = resolve_dtype(args.dtype, args.device)
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
    plot_rows(rows, args.out_dir, args.prefix, plot_dir)


if __name__ == "__main__":
    main()
