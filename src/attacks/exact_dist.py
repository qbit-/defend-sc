"""Exact-distribution log-likelihood scoring for SIPIT candidates.

For each candidate v with hypothetical residual diff = h(v) - obs, score the
candidate by log p(diff | distribution), where p is the *true* density of the
zero-mean noise eta drawn under the row's distribution and covariance Sigma.

This contrasts with the cov_only Mahalanobis score `-0.5 * diff^T Sigma^{-1} diff`,
which only matches the true log-likelihood when distribution == "gaussian".

Returned scores are higher-is-better and constants are dropped (they cancel in
argmax). The Rademacher branch is a *heuristic relaxation*; see comments below.
"""
from __future__ import annotations

import math

import torch

from ..noise import GaussianCov


_SQRT3 = math.sqrt(3.0)
_SQRT2 = math.sqrt(2.0)
_NEG_INF = -1e30  # Use a large negative finite value, not -inf, for arithmetic safety
                   # in case downstream code adds another finite term (it shouldn't here,
                   # but argmax is robust either way).


def score_diff_exact(diff: torch.Tensor, cov: GaussianCov,
                     distribution: str) -> torch.Tensor:
    """Higher = better. Log-likelihood of eta = diff under the row's distribution.

    Args:
        diff: tensor with shape [..., H]. Reduces over the last dim.
        cov:  the SAME GaussianCov used to draw eta; whitening uses Sigma^{-1/2}.
        distribution: one of {"gaussian", "uniform", "laplace", "rademacher"}.

    Returns:
        score with shape diff.shape[:-1].
    """
    if distribution == "gaussian":
        # Drop -0.5 * H * log(2*pi) - 0.5 * log|Sigma|; constant in argmax.
        return -0.5 * cov.maha(diff)

    if distribution == "uniform":
        # Each whitened coord z_i ~ Uniform(-sqrt(3), sqrt(3)) iid; density is the
        # constant 1/(2*sqrt(3))^d on the box, 0 outside. Inside -> log-likelihood
        # is a constant (drop it -> 0). Outside -> -inf.
        # NOTE: when sigma0 == 0 with low-rank present (lowrank_struct), `whiten`
        # already drops the perp component, so the feasibility test runs over the
        # rank-r in-span subspace only. That matches the noise's true support.
        z = cov.whiten(diff)
        feasible = (z.abs() <= _SQRT3 + 1e-5).all(dim=-1)
        return torch.where(feasible, torch.zeros_like(feasible, dtype=diff.dtype),
                           torch.full_like(feasible, _NEG_INF, dtype=diff.dtype))

    if distribution == "laplace":
        # Standard Laplace with unit variance has scale b = 1/sqrt(2):
        #   p(z) = (1 / (2b)) exp(-|z| / b) = (1 / sqrt(2)) exp(-sqrt(2) |z|).
        # Sum log p over coords; drop the constant.
        z = cov.whiten(diff)
        return -_SQRT2 * z.abs().sum(dim=-1)

    if distribution == "rademacher":
        # Heuristic relaxation: Rademacher's true support is {-1, +1}^H so the
        # density is a sum of point masses; continuous candidate residuals
        # almost never land *exactly* on a vertex, so the true log-likelihood
        # is -inf almost everywhere. We instead score by squared distance from
        # the nearest vertex in whitened space, i.e. sum_i (z_i - sign(z_i))^2.
        z = cov.whiten(diff)
        return -((z - z.sign()) ** 2).sum(dim=-1)

    raise ValueError(
        f"unknown distribution {distribution!r}; "
        f"expected one of {{'gaussian', 'uniform', 'laplace', 'rademacher'}}"
    )
