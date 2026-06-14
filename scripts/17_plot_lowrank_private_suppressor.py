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


def load_one_shot(csv_path: Path) -> tuple[list[float], list[dict], list[dict], float]:
    rows = list(csv.DictReader(csv_path.open()))
    by = {}
    for row in rows:
        if row["repeats"] == "1" and row["repeat_policy"] in ("independent_average", "clean_reference"):
            by[(float(row["sigma0_frac"]), row["variant"])] = row
    sfs = sorted(sf for sf, variant in by if variant == "b1_lowrank_struct")
    raw = [by[(sf, "b1_lowrank_struct")] for sf in sfs]
    private = [by[(sf, "b1_lowrank_struct_private_suppressor")] for sf in sfs]
    clean = max(float(row["acc_clean_test"]) for row in rows if row.get("acc_clean_test"))
    return sfs, raw, private, clean


def plot_all(csv_path: Path, prefix: str) -> list[Path]:
    sfs, raw, private, clean = load_one_shot(csv_path)
    plot_dir = util.ART / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = []

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(sfs, [fnum(row, "acc_with_corr") for row in raw], "o-",
            label="raw lowrank_struct", alpha=0.75)
    ax.plot(sfs, [fnum(row, "acc_with_corr") for row in private], "s-",
            label="server-private suppressor", alpha=0.75)
    ax.axhline(clean, color="0.25", linestyle=":", linewidth=1.4,
               label=f"clean baseline ({clean:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("noise scale sf")
    ax.set_ylabel("SST-2 accuracy")
    ax.set_title("Lowrank-Struct Utility With Server-Private Suppression")
    ax.set_ylim(0.43, 0.93)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    out.append(save(fig, plot_dir, prefix, "utility_vs_scale"))

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for col, label in [
        ("eve_raw_exact", "exact/Mahalanobis"),
        ("eve_seq_map", "seq-MAP"),
        ("eve_vanilla", "vanilla"),
    ]:
        ax.plot(sfs, [fnum(row, col) for row in raw], "o-", label=label, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("noise scale sf")
    ax.set_ylabel("Eve token top-1")
    ax.set_title("Raw One-Shot Eve Recovery For Lowrank-Struct Noise")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    out.append(save(fig, plot_dir, prefix, "eve_vs_scale"))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    exact = [fnum(row, "eve_raw_exact") for row in raw]
    raw_acc = [fnum(row, "acc_with_corr") for row in raw]
    private_acc = [fnum(row, "acc_with_corr") for row in private]
    ax.plot(exact, raw_acc, "o-", label="raw lowrank_struct utility", alpha=0.5)
    ax.plot(exact, private_acc, "s-", label="server-private suppressed utility", alpha=0.5)
    for x, y, sf in zip(exact, private_acc, sfs):
        if sf in (0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0):
            ax.annotate(f"{sf:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(clean, color="0.25", linestyle=":", linewidth=1.4,
               label=f"clean baseline ({clean:.3f})")
    ax.set_xlabel("exact Eve token top-1, m=1")
    ax.set_ylabel("SST-2 accuracy")
    ax.set_title("One-Shot Privacy / Utility Frontier")
    ax.set_ylim(0.43, 0.93)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    out.append(save(fig, plot_dir, prefix, "exact_privacy_utility_frontier"))

    fig, ax1 = plt.subplots(figsize=(8.2, 4.9))
    gain = [private_y - raw_y for private_y, raw_y in zip(private_acc, raw_acc)]
    distortion = [fnum(row, "clean_distortion") for row in private]
    ax1.plot(sfs, gain, "o-", color="#2ca02c", label="accuracy gain", alpha=0.75)
    ax1.axhline(0.0, color="0.55", linewidth=1.0)
    ax1.set_xscale("log")
    ax1.set_xlabel("noise scale sf")
    ax1.set_ylabel("private acc - raw acc")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(sfs, distortion, "s--", color="#d62728", label="clean distortion", alpha=0.65)
    ax2.set_ylabel("clean activation distortion")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_title("Server-Private Suppressor Gain vs Clean Distortion")
    out.append(save(fig, plot_dir, prefix, "gain_and_clean_distortion"))

    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=util.ART / "setting_g_qwen_sst2_lowrank_private_suppressor" / "tnsc_eval.csv",
    )
    ap.add_argument("--prefix", default="setting_g_qwen_sst2_lowrank_private_suppressor")
    return ap.parse_args()


def main():
    args = parse_args()
    plot_all(args.csv, args.prefix)


if __name__ == "__main__":
    main()
