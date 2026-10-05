"""Plot lowrank_struct server-private suppressor results."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

from src import util


def fnum(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def save(fig, plot_dir: Path, prefix: str, name: str) -> Path:
    path = plot_dir / f"{prefix}_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    return path


def load_one_shot(csv_path: Path) -> tuple[list[float], list[dict], list[dict], list[dict]]:
    """Pair raw and suppressed rows at repeats=1.

    Args:
        csv_path: ``tnsc_eval.csv``.

    Returns:
        Noise scales, raw rows, suppressed rows, and every CSV row.
    """
    rows = list(csv.DictReader(csv_path.open()))
    by = {}
    for row in rows:
        if row["repeats"] == "1" and row["repeat_policy"] in (
            "independent_average", "clean_reference",
        ):
            by[(float(row["sigma0_frac"]), row["variant"])] = row
    sfs = sorted(sf for sf, variant in by if variant == "b1_lowrank_struct")
    raw = [by[(sf, "b1_lowrank_struct")] for sf in sfs]
    private = [
        by[(sf, "b1_lowrank_struct_private_suppressor")] for sf in sfs
    ]
    return sfs, raw, private, rows


def metric_names(rows: list[dict]) -> list[str]:
    """Return utility metrics that have a raw or private column filled.

    Args:
        rows: CSV rows.

    Returns:
        Metric ids in plot order.
    """
    known = ("accuracy", "exact_match", "char_edit_similarity")
    found = []
    for name in known:
        raw_key = f"{name}_raw"
        private_key = f"{name}_private"
        if any(row.get(raw_key) not in ("", None) for row in rows):
            found.append(name)
        elif any(row.get(private_key) not in ("", None) for row in rows):
            found.append(name)
    return found


def metric_label(name: str) -> str:
    """Return the y-axis label for a utility metric.

    Args:
        name: Metric id.

    Returns:
        Human-readable axis label.
    """
    labels = {
        "accuracy": "Accuracy",
        "exact_match": "Exact match",
        "char_edit_similarity": "Character edit similarity",
    }
    return labels.get(name, name)


def y_limits(values: list[float | None]) -> tuple[float, float]:
    """Pad observed scores into a range inside ``[0, 1]``.

    Args:
        values: Scores, with ``None`` for missing points.

    Returns:
        Lower and upper y limits.
    """
    present = [value for value in values if value is not None]
    if not present:
        return 0.0, 1.0
    low = min(present)
    high = max(present)
    pad = max(0.05, 0.1 * (high - low))
    return max(0.0, low - pad), min(1.0, high + pad)


def _clean_value(rows: list[dict], metric: str) -> float | None:
    """Return the clean baseline for ``metric``.

    Args:
        rows: Raw-channel rows.
        metric: Utility metric id.

    Returns:
        The first non-empty clean score, or ``None``.
    """
    for row in rows:
        value = fnum(row, f"{metric}_clean")
        if value is not None:
            return value
    return None


def _series(rows: list[dict], metric: str, channel: str) -> list[float | None]:
    """Read one channel of a utility metric.

    Args:
        rows: Aligned eval rows.
        metric: Utility metric id.
        channel: ``raw`` or ``private``.

    Returns:
        One score per row.
    """
    return [fnum(row, f"{metric}_{channel}") for row in rows]


def _plot_utility(
    sfs, raw_y, private_y, clean, limits, label, plot_dir, prefix, metric,
):
    """Write utility versus noise scale.

    Args:
        sfs: Noise scales.
        raw_y: Raw utility values.
        private_y: Suppressed utility values.
        clean: Clean baseline, or ``None``.
        limits: Y-axis limits.
        label: Axis label.
        plot_dir: Figure directory.
        prefix: Filename prefix.
        metric: Metric id used in the filename.

    Returns:
        Written PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(sfs, raw_y, "o-", label="raw lowrank_struct", alpha=0.75)
    ax.plot(
        sfs, private_y, "s-",
        label="server-private suppressor", alpha=0.75,
    )
    if clean is not None:
        ax.axhline(
            clean, color="0.25", linestyle=":", linewidth=1.4,
            label=f"clean baseline ({clean:.3f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("noise scale sf")
    ax.set_ylabel(label)
    ax.set_title(f"Lowrank-Struct {label}")
    ax.set_ylim(*limits)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    return save(fig, plot_dir, prefix, f"utility_vs_scale_{metric}")


def _plot_frontier(
    sfs, raw, raw_y, private_y, clean, limits, label, plot_dir, prefix, metric,
):
    """Write utility against Eve recovery.

    Args:
        sfs: Noise scales.
        raw: Raw-channel rows, used for Eve scores.
        raw_y: Raw utility values.
        private_y: Suppressed utility values.
        clean: Clean baseline, or ``None``.
        limits: Y-axis limits.
        label: Axis label.
        plot_dir: Figure directory.
        prefix: Filename prefix.
        metric: Metric id used in the filename.

    Returns:
        Written PNG path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    exact = [fnum(row, "eve_raw_exact") for row in raw]
    ax.plot(exact, raw_y, "o-", label="raw lowrank_struct utility", alpha=0.5)
    ax.plot(
        exact, private_y, "s-",
        label="server-private suppressed utility", alpha=0.5,
    )
    marked_scales = (0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0)
    for x_value, y_value, sf in zip(exact, private_y, sfs):
        if sf in marked_scales and y_value is not None:
            ax.annotate(
                f"{sf:g}", (x_value, y_value),
                textcoords="offset points", xytext=(4, 4), fontsize=8,
            )
    if clean is not None:
        ax.axhline(
            clean, color="0.25", linestyle=":", linewidth=1.4,
            label=f"clean baseline ({clean:.3f})",
        )
    ax.set_xlabel("exact Eve token top-1, m=1")
    ax.set_ylabel(label)
    ax.set_title(f"Privacy / Utility Frontier ({label})")
    ax.set_ylim(*limits)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    return save(fig, plot_dir, prefix, f"frontier_{metric}")


def _plot_gain(sfs, raw_y, private_y, private, label, plot_dir, prefix, metric):
    """Write suppressor gain and clean distortion.

    Args:
        sfs: Noise scales.
        raw_y: Raw utility values.
        private_y: Suppressed utility values.
        private: Suppressed rows, used for distortion.
        label: Axis label.
        plot_dir: Figure directory.
        prefix: Filename prefix.
        metric: Metric id used in the filename.

    Returns:
        Written PNG path.
    """
    fig, ax1 = plt.subplots(figsize=(8.2, 4.9))
    gain = [
        None if left is None or right is None else right - left
        for left, right in zip(raw_y, private_y)
    ]
    distortion = [fnum(row, "clean_distortion") for row in private]
    ax1.plot(
        sfs, gain, "o-", color="#2ca02c", label=f"{label} gain", alpha=0.75,
    )
    ax1.axhline(0.0, color="0.55", linewidth=1.0)
    ax1.set_xscale("log")
    ax1.set_xlabel("noise scale sf")
    ax1.set_ylabel(f"private {label} - raw {label}")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        sfs, distortion, "s--", color="#d62728",
        label="clean distortion", alpha=0.65,
    )
    ax2.set_ylabel("clean activation distortion")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left",
    )
    ax1.set_title(f"Suppressor Gain vs Clean Distortion ({label})")
    return save(fig, plot_dir, prefix, f"gain_{metric}")


def plot_metric(
    sfs, raw, private, metric: str, plot_dir: Path, prefix: str,
) -> list[Path]:
    """Write utility, frontier, and gain plots for one metric.

    Args:
        sfs: Noise scales.
        raw: Raw-channel rows.
        private: Suppressed-channel rows.
        metric: Utility metric id.
        plot_dir: Directory under ``artifacts/plots``.
        prefix: Filename prefix.

    Returns:
        Written PNG paths.
    """
    label = metric_label(metric)
    raw_y = _series(raw, metric, "raw")
    private_y = _series(private, metric, "private")
    clean = _clean_value(raw, metric)
    limits = y_limits(
        [value for value in raw_y + private_y + [clean] if value is not None]
    )
    return [
        _plot_utility(
            sfs, raw_y, private_y, clean, limits, label,
            plot_dir, prefix, metric,
        ),
        _plot_frontier(
            sfs, raw, raw_y, private_y, clean, limits, label,
            plot_dir, prefix, metric,
        ),
        _plot_gain(
            sfs, raw_y, private_y, private, label, plot_dir, prefix, metric,
        ),
    ]


def plot_eve(sfs, raw, plot_dir: Path, prefix: str) -> Path:
    """Write the Eve-versus-scale plot.

    Args:
        sfs: Noise scales.
        raw: Raw-channel rows.
        plot_dir: Figure directory.
        prefix: Filename prefix.

    Returns:
        Written PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for col, label in [
        ("eve_raw_exact", "exact/Mahalanobis"),
        ("eve_seq_map", "seq-MAP"),
        ("eve_vanilla", "vanilla"),
    ]:
        ax.plot(
            sfs, [fnum(row, col) for row in raw],
            "o-", label=label, alpha=0.75,
        )
    ax.set_xscale("log")
    ax.set_xlabel("noise scale sf")
    ax.set_ylabel("Eve token top-1")
    ax.set_title("Raw One-Shot Eve Recovery For Lowrank-Struct Noise")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    return save(fig, plot_dir, prefix, "eve_vs_scale")


def _task_from_eval_csv(csv_path: Path) -> str:
    """Read the benchmark name from an eval CSV path.

    Args:
        csv_path: ``artifacts/evals/<model>/<task>/tnsc_eval.csv``.

    Returns:
        The task directory name.

    Raises:
        SystemExit: If the path is not under ``evals/<model>/<task>``.
    """
    parts = csv_path.resolve().parts
    if "evals" not in parts:
        raise SystemExit(
            "csv must be artifacts/evals/<model>/<task>/tnsc_eval.csv"
        )
    index = parts.index("evals")
    if index + 2 >= len(parts):
        raise SystemExit(
            "csv must be artifacts/evals/<model>/<task>/tnsc_eval.csv"
        )
    return parts[index + 2]


def plot_all(csv_path: Path, prefix: str) -> list[Path]:
    """Write one figure set per utility metric, plus the Eve plot.

    Args:
        csv_path: ``tnsc_eval.csv``. The parent folders name the model
            and benchmark when the file lives under ``artifacts/evals``.
        prefix: Filename prefix.

    Returns:
        Written PNG paths.
    """
    sfs, raw, private, rows = load_one_shot(csv_path)
    model_id = rows[0]["model_id"] if rows else "unknown"
    task_name = _task_from_eval_csv(csv_path)
    plot_dir = util.art_path("plots", model_id, task_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = [plot_eve(sfs, raw, plot_dir, prefix)]
    for metric in metric_names(rows):
        out.extend(plot_metric(sfs, raw, private, metric, plot_dir, prefix))
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
    )
    ap.add_argument("--prefix", default="lowrank_private_suppressor")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.csv is None:
        raise SystemExit(
            "pass --csv artifacts/evals/<model>/<task>/tnsc_eval.csv"
        )
    plot_all(args.csv, args.prefix)


if __name__ == "__main__":
    main()
