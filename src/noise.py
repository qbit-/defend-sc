"""Gaussian covariance for cut activations (singular low-rank + isotropic floor).

Covariance representation: Sigma = sigma0^2 I + U Lambda U^T (low-rank correction
on top of an isotropic floor). Sampling:
  eta = sigma0 * z0 + U @ diag(sqrt(Lambda)) @ z1
where z0, z1 are unit-variance iid factors drawn from the requested distribution
(`gaussian | uniform | laplace | rademacher`). The covariance Sigma is identical
across distributions; only the *factor* z changes. This is why covariance
artifact files are keyed by (sigma0, family) but NOT by distribution.
"""
from __future__ import annotations
import math
import torch
from dataclasses import dataclass


SUPPORTED_DISTRIBUTIONS = ("gaussian", "uniform", "laplace", "rademacher")


def _sample_unit_variance(shape, distribution: str, generator: torch.Generator,
                          dtype=torch.float32) -> torch.Tensor:
    """Draw iid samples with mean 0 and variance 1 from `distribution`.

    Implementation lives outside `GaussianCov` so it can be reused by tests.
    Always sampled on CPU with the supplied torch.Generator for portability.
    """
    if distribution == "gaussian":
        return torch.randn(*shape, generator=generator, dtype=dtype)
    if distribution == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] has variance 1.
        bound = math.sqrt(3.0)
        u = torch.rand(*shape, generator=generator, dtype=dtype)
        return u.mul_(2.0 * bound).sub_(bound)
    if distribution == "laplace":
        # Laplace(0, b) has variance 2 b^2; b = 1/sqrt(2) -> variance 1.
        # Inverse-CDF: sign(U-0.5) * b * ln(1 - 2|U-0.5|)
        b = 1.0 / math.sqrt(2.0)
        u = torch.rand(*shape, generator=generator, dtype=dtype)
        u.sub_(0.5)
        # clamp away from +/-0.5 to avoid log(0)
        eps = torch.finfo(dtype).tiny
        sign = u.sign()
        # 1 - 2|u| in (0, 1]; clamp lower by eps
        mag = (1.0 - 2.0 * u.abs()).clamp_min_(eps)
        return sign.mul_(-b).mul_(mag.log_())
    if distribution == "rademacher":
        # +1/-1 with equal probability; variance is exactly 1.
        u = torch.rand(*shape, generator=generator, dtype=dtype)
        return torch.where(u < 0.5,
                           torch.full_like(u, -1.0),
                           torch.full_like(u, 1.0))
    raise ValueError(f"unknown distribution {distribution!r}; "
                     f"expected one of {SUPPORTED_DISTRIBUTIONS}")


def whiten_low_rank(x: torch.Tensor, sigma0: float,
                    U: torch.Tensor | None, lam: torch.Tensor | None
                    ) -> torch.Tensor:
    """Apply Sigma^{-1/2} to x for a low-rank+isotropic Sigma.

    Sigma = sigma0^2 * I + U diag(lam) U^T (U column-orthonormal, lam >= 0).

    Pure-tensor function: no `GaussianCov` allocation, autograd-friendly. Used
    by both `GaussianCov.whiten` and the CW collision objective in
    `scripts/10_min_collision.py` (where autograd through `x` must be preserved).
    See `GaussianCov.whiten` docstring for the three regime cases.
    """
    s2 = float(sigma0) ** 2
    if U is None or U.shape[1] == 0:
        if sigma0 == 0:
            return torch.zeros_like(x)
        return x / sigma0
    if sigma0 == 0:
        proj = x @ U
        inv_sqrt_lam = lam.clamp(min=1e-12).rsqrt()
        return (proj * inv_sqrt_lam) @ U.T
    a = (s2 + lam).sqrt()
    b = float(s2 ** 0.5)
    proj = x @ U
    coeff = (1.0 / a - 1.0 / b)
    return x / b + (proj * coeff) @ U.T


@dataclass
class GaussianCov:
    name: str
    hidden: int
    sigma0: float                       # isotropic floor
    U: torch.Tensor | None = None       # [hidden, r] low-rank factor
    lam: torch.Tensor | None = None     # [r] eigenvalues (variance contribution)
    note: str = ""

    @property
    def rank(self) -> int:
        return 0 if self.U is None else self.U.shape[1]

    def to(self, device, dtype=None):
        out = GaussianCov(self.name, self.hidden, self.sigma0,
                         note=self.note)
        if self.U is not None:
            out.U = self.U.to(device=device, dtype=dtype) if dtype else self.U.to(device)
            out.lam = self.lam.to(device=device, dtype=dtype) if dtype else self.lam.to(device)
        return out

    def sample(self, shape, generator: torch.Generator, device, dtype=torch.float32,
               distribution: str = "gaussian"):
        """Sample iid noise. shape = (..., hidden).
        """
        z0 = _sample_unit_variance(shape, distribution, generator,
                                   dtype=torch.float32)
        z0 = z0.to(device=device, dtype=dtype)
        eta = self.sigma0 * z0
        if self.U is not None and self.U.shape[1] > 0:
            r = self.U.shape[1]
            lead = tuple(shape[:-1]) + (r,)
            z1 = _sample_unit_variance(lead, distribution, generator,
                                       dtype=torch.float32)
            z1 = z1.to(device=device, dtype=dtype)
            U = self.U.to(device=device, dtype=dtype)
            lam = self.lam.to(device=device, dtype=dtype)
            eta = eta + (z1 * lam.sqrt()) @ U.T
        return eta

    def diag(self) -> torch.Tensor:
        """Diagonal of Sigma (per-coordinate variance)."""
        d = torch.full((self.hidden,), self.sigma0 ** 2)
        if self.U is not None:
            d = d + (self.U.float() ** 2 * self.lam.float().unsqueeze(0)).sum(dim=1)
        return d

    def trace(self) -> float:
        return float(self.diag().sum())

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sigma^{-1/2} x for Mahalanobis use.

        Three regimes:
          - sigma0 > 0, no low-rank: Sigma = sigma0^2 I, whiten = x / sigma0.
          - sigma0 > 0, low-rank present: orthonormal U with eigvals (sigma0^2 + lam) on
            span(U) and sigma0^2 elsewhere; Sigma^{-1/2} via in-span/perp split.
          - sigma0 == 0 with low-rank present (lowrank_struct family): Sigma is
            singular outside span(U). Perp directions are treated as
            infinite-precision and dropped from the maha distance.
        """
        if self.U is None or self.U.shape[1] == 0:
            return whiten_low_rank(x, self.sigma0, None, None)
        U = self.U.to(x.device, x.dtype)
        lam = self.lam.to(x.device, x.dtype)
        return whiten_low_rank(x, self.sigma0, U, lam)

    def maha(self, x: torch.Tensor) -> torch.Tensor:
        """Return ||Sigma^{-1/2} x||^2 along last dim."""
        w = self.whiten(x)
        return (w * w).sum(dim=-1)
