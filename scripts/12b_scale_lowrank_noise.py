"""Prepare low-amplitude B1 covariance artifacts for Qwen lowrank_struct probes.

The existing Setting-B1 grid starts at sf=0.5. For Qwen/Qwen2.5-0.5B that is
already too large for utility, especially at k=8. This helper creates lower
sf covariance files by reusing an existing covariance's orientation and scaling
its variance:

    sigma0 <- sigma0 * (target_sf / base_sf)
    lam    <- lam    * (target_sf / base_sf)^2

For lowrank_struct from 04_build_covariance.py, sigma0 is zero and U is
independent of sf, so this is equivalent to rebuilding the same structured
covariance at lower total variance without recomputing geometry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import util
from src import seeding


def cov_path(
    model_id: str, task: str, k: int, sf: float, rank: int, family: str,
) -> Path:
    """Return one covariance artifact path.

    Args:
        model_id: Hugging Face model id.
        task: Benchmark name.
        k: Split depth.
        sf: Noise scale.
        rank: Covariance rank.
        family: Covariance family name.

    Returns:
        Path of the ``.pt`` covariance file.
    """
    return util.art_path(
        "covariances", model_id, task, f"split_{k}",
        f"sigma0_{sf:g}_r{rank}__{family}.pt",
    )


def scale_covariance_blob(blob: dict, family: str, base_sf: float,
                          target_sf: float, rank: int) -> dict:
    ratio = float(target_sf) / float(base_sf)
    lam = blob.get("lam")
    scaled_lam = None if lam is None else lam.float() * (ratio ** 2)
    note = blob.get("note", "")
    scaled_note = (
        f"{note}; variance-scaled from sf={base_sf:g} by "
        f"(target/base)^2={ratio ** 2:.6g}"
    ).strip("; ")
    return {
        "U": blob.get("U"),
        "lam": scaled_lam,
        "sigma0": float(blob["sigma0"]) * ratio,
        "name": f"{family}_scaled_{target_sf:g}_r{rank}",
        "hidden": int(blob["hidden"]),
        "note": scaled_note,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--families", nargs="+", default=["lowrank_struct"])
    ap.add_argument("--base-sf", type=float, default=0.5)
    ap.add_argument("--target-sfs", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.35])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    seeding.set_seed(args.seed)
    written = []
    for model_id in args.models:
        for k in args.ks:
            for family in args.families:
                src = cov_path(
                    model_id, args.task, k, args.base_sf, args.rank, family,
                )
                if not src.exists():
                    raise FileNotFoundError(
                        f"missing base covariance {src}; run 04_build_covariance.py "
                        "or copy the artifacts before scaling"
                    )
                blob = torch.load(src, weights_only=True)
                for sf in args.target_sfs:
                    dst = cov_path(
                        model_id, args.task, k, sf, args.rank, family,
                    )
                    if dst.exists() and not args.overwrite:
                        print(f"skip existing {dst}")
                        continue
                    scaled = scale_covariance_blob(blob, family, args.base_sf,
                                                   sf, args.rank)
                    if args.dry_run:
                        print(f"would write {dst}")
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(scaled, dst)
                        print(f"wrote {dst}")
                    written.append(dst)
    print(f"scaled {len(written)} covariance files", flush=True)


if __name__ == "__main__":
    main()
