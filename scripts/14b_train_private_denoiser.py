"""Train the lowrank_struct server-side private suppressor per noise scale."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

from src import util
from src import seeding
from src.private_denoise import PrivateLowrankStructSuppressor


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required to run this script outside --help")
    return torch


def _no_grad():
    return (lambda fn: fn) if torch is None else torch.no_grad()


def model_safe(model_id: str) -> str:
    return model_id.replace("/", "_")


def cache_path(model_id: str, task: str, k: int) -> Path:
    return util.ART / "activations" / model_safe(model_id) / task / f"split_{k}" / "cache.pt"


def cov_path(model_id: str, task: str, k: int, sigma0_frac: float, rank: int,
             family: str) -> Path:
    return (
        util.ART / "covariances" / model_safe(model_id) / task / f"split_{k}"
        / f"sigma0_{sigma0_frac:g}_r{rank}__{family}.pt"
    )


def private_suffix(family: str) -> str:
    if family == "lowrank_struct":
        return "lowrank_struct_private_suppressor"
    return family


def out_dir(model_id: str, task: str, k: int, sigma0_frac: float, rank: int,
            family: str) -> Path:
    return (
        util.ART / "private_denoisers" / model_safe(model_id) / task / f"split_{k}"
        / f"sigma0_{sigma0_frac:g}_r{rank}__{private_suffix(family)}"
    )


def _load_runtime_modules():
    from src import noise as N

    return N


def load_cov(N, path: Path):
    blob = torch.load(path, weights_only=True)
    return N.GaussianCov(
        name=blob["name"],
        hidden=blob["hidden"],
        sigma0=blob["sigma0"],
        U=blob["U"],
        lam=blob["lam"],
        note=blob["note"],
    ), blob


def _valid_flat(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    H = x.shape[-1]
    return x.reshape(-1, H).float()[mask.bool().reshape(-1)]


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean())


@_no_grad()
def train_one(
    model_id: str,
    task: str,
    k: int,
    sigma0_frac: float,
    rank: int,
    family: str,
    seed: int = 0,
):
    torch = _require_torch()
    N = _load_runtime_modules()
    safe = model_safe(model_id)
    print(
        f"\n--- private denoiser {task} {model_id} k={k} "
        f"sf={sigma0_frac:g} r={rank} family={family} ---",
        flush=True,
    )

    cache_p = cache_path(model_id, task, k)
    covariance_p = cov_path(model_id, task, k, sigma0_frac, rank, family)
    if not cache_p.exists():
        print(f"  skip: missing activation cache {cache_p}", flush=True)
        return None
    if not covariance_p.exists():
        print(f"  skip: missing TNSC covariance {covariance_p}", flush=True)
        return None

    cache = torch.load(cache_p, weights_only=False)
    train = cache["train"]
    if "clipped_a" not in train:
        print("  skip: cache['train'] is missing clipped_a", flush=True)
        return None
    if "mask" not in train:
        print("  skip: cache['train'] is missing mask", flush=True)
        return None

    clipped_a = train["clipped_a"].cpu().float()
    mask = train["mask"].cpu().bool()
    cov, cov_blob = load_cov(N, covariance_p)
    if int(cov.hidden) != int(clipped_a.shape[-1]):
        raise ValueError(
            f"covariance hidden={cov.hidden} does not match activations H={clipped_a.shape[-1]}"
        )

    if family != "lowrank_struct":
        raise ValueError(f"only family 'lowrank_struct' is supported, got {family!r}")
    denoiser = PrivateLowrankStructSuppressor.fit(
        calibration_a=clipped_a,
        mask=mask,
        cov_eta=cov,
    )
    denoiser_method = "private_lowrank_struct_suppressor"

    clean_valid = _valid_flat(clipped_a, mask)
    denoised_clean_valid = _valid_flat(denoiser.posterior_mean(clipped_a), mask)
    clean_distortion = _relative_l2(denoised_clean_valid, clean_valid)
    clean_mse = _mse(denoised_clean_valid, clean_valid)

    gen = torch.Generator().manual_seed(int(seed))
    eta = cov.sample(clipped_a.shape, gen, device="cpu", dtype=torch.float32, distribution="gaussian")
    noisy = clipped_a + eta
    noisy_valid = _valid_flat(noisy, mask)
    denoised_noisy_valid = _valid_flat(denoiser.posterior_mean(noisy), mask)
    raw_noisy_distortion = _relative_l2(noisy_valid, clean_valid)
    denoised_noisy_distortion = _relative_l2(denoised_noisy_valid, clean_valid)
    raw_noisy_mse = _mse(noisy_valid, clean_valid)
    denoised_noisy_mse = _mse(denoised_noisy_valid, clean_valid)

    path = out_dir(model_id, task, k, sigma0_frac, rank, family)
    path.mkdir(parents=True, exist_ok=True)
    blob = {
        **denoiser.state_dict(),
        "model_id": model_id,
        "model_safe": safe,
        "task": task,
        "k": int(k),
        "sigma0_frac": float(sigma0_frac),
        "rank": int(rank),
        "family": family,
        "private_suffix": private_suffix(family),
        "denoiser_method": denoiser_method,
        "covariance_path": str(covariance_p),
        "covariance_name": cov_blob.get("name"),
        "covariance_note": cov_blob.get("note"),
    }
    torch.save(blob, path / "private_denoiser.pt")

    metrics = {
        "meta": util.base_meta(
            Path(__file__).name,
            model_id=model_id,
            model_safe=safe,
            task=task,
            k=k,
            sigma0_frac=sigma0_frac,
            rank=rank,
            seed=seed,
        ),
        "model_id": model_id,
        "model_safe": safe,
        "task": task,
        "k": int(k),
        "sigma0_frac": float(sigma0_frac),
        "rank": int(rank),
        "family": family,
        "private_suffix": private_suffix(family),
        "denoiser_method": denoiser_method,
        "hidden": int(denoiser.hidden),
        "n_valid_train_positions": int(mask.sum()),
        "mean_gamma": float(denoiser.gamma.mean()) if denoiser.gamma.numel() else 0.0,
        "max_gamma": float(denoiser.gamma.max()) if denoiser.gamma.numel() else 0.0,
        "sigma0": float(cov.sigma0),
        "noise_rank": int(cov.rank),
        "noise_trace": float(cov.trace()),
        "clean_reconstruction_distortion": clean_distortion,
        "clean_reconstruction_mse": clean_mse,
        "raw_noisy_distortion": raw_noisy_distortion,
        "denoised_noisy_distortion": denoised_noisy_distortion,
        "raw_noisy_mse": raw_noisy_mse,
        "denoised_noisy_mse": denoised_noisy_mse,
        "mse_improvement": raw_noisy_mse / max(denoised_noisy_mse, 1e-30),
        "covariance_path": str(covariance_p),
        "artifact_path": str(path / "private_denoiser.pt"),
    }
    util.write_json(path / "metrics.json", metrics)
    print(
        f"  wrote {path / 'private_denoiser.pt'} "
        f"D(clean)={clean_distortion:.3g} "
        f"D(noisy)={denoised_noisy_distortion:.3g} "
        f"raw={raw_noisy_distortion:.3g}",
        flush=True,
    )
    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--sfs", nargs="+", type=float, default=[0.5, 1.5, 3.0, 6.0])
    ap.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--families", nargs="+", default=["lowrank_struct"],
                    choices=["lowrank_struct"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    _require_torch()
    seeding.set_seed(args.seed)
    rows = []
    for mid in args.models:
        for k in args.ks:
            for sf in args.sfs:
                for rank in args.ranks:
                    for family in args.families:
                        row = train_one(
                            mid,
                            task=args.task,
                            k=k,
                            sigma0_frac=sf,
                            rank=rank,
                            family=family,
                            seed=args.seed,
                        )
                        if row is not None:
                            rows.append(row)
    util.write_json(util.ART / "private_denoiser_summary.json", {"rows": rows})


if __name__ == "__main__":
    main()
