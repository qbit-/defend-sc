"""Private server-side suppressor for lowrank_struct TNSC activations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _valid_rows(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    H = a.shape[-1]
    m = mask.bool().reshape(-1)
    X = a.reshape(-1, H).float()[m]
    if X.numel() == 0:
        raise ValueError("no valid activation rows found after applying mask")
    return X


@dataclass
class PrivateLowrankStructSuppressor:
    """Server-private Wiener shrinker for singular lowrank_struct noise.

    Noise only lives in span(U_eta), so the denoiser leaves the perpendicular
    coordinates untouched and shrinks only the noised coordinates toward their
    calibration mean.
    """

    mean: torch.Tensor
    U_eta: torch.Tensor
    lam_eta: torch.Tensor
    prior_var: torch.Tensor
    gamma: torch.Tensor
    hidden: int

    @classmethod
    def fit(
        cls,
        calibration_a: torch.Tensor,
        mask: torch.Tensor,
        cov_eta,
        min_prior_var: float = 1e-8,
    ) -> "PrivateLowrankStructSuppressor":
        if cov_eta.U is None or cov_eta.U.numel() == 0:
            raise ValueError("lowrank_struct suppressor requires a non-empty noise subspace")
        X = _valid_rows(calibration_a, mask)
        mean = X.mean(dim=0).float()
        U_eta = cov_eta.U.float().contiguous()
        lam_eta = cov_eta.lam.float().contiguous()
        centered = X - mean
        coeff = centered @ U_eta
        if coeff.shape[0] <= 1:
            prior_var = torch.full_like(lam_eta, float(min_prior_var))
        else:
            prior_var = coeff.var(dim=0, unbiased=True).clamp(min=float(min_prior_var))
        gamma = (lam_eta / (prior_var + lam_eta).clamp(min=1e-30)).clamp(0.0, 1.0)
        return cls(
            mean=mean,
            U_eta=U_eta,
            lam_eta=lam_eta,
            prior_var=prior_var.float(),
            gamma=gamma.float(),
            hidden=int(mean.numel()),
        )

    def to(self, device=None, dtype=None) -> "PrivateLowrankStructSuppressor":
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return PrivateLowrankStructSuppressor(
            mean=self.mean.to(**kwargs),
            U_eta=self.U_eta.to(**kwargs),
            lam_eta=self.lam_eta.to(**kwargs),
            prior_var=self.prior_var.to(**kwargs),
            gamma=self.gamma.to(**kwargs),
            hidden=self.hidden,
        )

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.hidden:
            raise ValueError(f"expected last dim {self.hidden}, got {y.shape[-1]}")
        if self.U_eta.numel() == 0:
            return y
        mean = self.mean.to(y.device, y.dtype)
        U = self.U_eta.to(y.device, y.dtype)
        gamma = self.gamma.to(y.device, y.dtype)
        centered = y - mean
        coeff = centered @ U
        return y - (coeff * gamma) @ U.T

    def posterior_mean(self, y: torch.Tensor) -> torch.Tensor:
        return self.apply(y)

    def state_dict(self) -> dict:
        return {
            "format_version": 1,
            "denoiser_type": "private_lowrank_struct_suppressor",
            "mean": self.mean.cpu(),
            "mu": self.mean.cpu(),
            "U_eta": self.U_eta.cpu(),
            "lam_eta": self.lam_eta.cpu(),
            "prior_var": self.prior_var.cpu(),
            "gamma": self.gamma.cpu(),
            "hidden": int(self.hidden),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PrivateLowrankStructSuppressor":
        mean = state.get("mean", state.get("mu"))
        return cls(
            mean=mean.float(),
            U_eta=state["U_eta"].float(),
            lam_eta=state["lam_eta"].float(),
            prior_var=state["prior_var"].float(),
            gamma=state["gamma"].float(),
            hidden=int(state["hidden"]),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PrivateLowrankStructSuppressor":
        blob = torch.load(path, weights_only=True)
        return cls.from_state_dict(blob)
