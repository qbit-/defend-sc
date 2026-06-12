"""Binary classification metrics for SST-2 sentiment."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """logits [B, n_classes]; labels [B]."""
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean())


def kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) averaged across batch. logits [B, n_classes]."""
    p = p_logits.float(); q = q_logits.float()
    log_p = F.log_softmax(p, dim=-1)
    log_q = F.log_softmax(q, dim=-1)
    p_prob = log_p.exp()
    return float((p_prob * (log_p - log_q)).sum(dim=-1).mean())


def top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    return float((p_logits.argmax(-1) == q_logits.argmax(-1)).float().mean())


def _assert_bkc_logits(clean_logits: torch.Tensor, noisy_logits: torch.Tensor):
    assert clean_logits.dim() == 2, "clean_logits must be [B, C]"
    assert noisy_logits.dim() == 3, "noisy_logits must be [B, K, C]"
    assert clean_logits.shape[0] == noisy_logits.shape[0], (
        f"batch mismatch: clean B={clean_logits.shape[0]} noisy B={noisy_logits.shape[0]}"
    )
    assert clean_logits.shape[1] == noisy_logits.shape[2], (
        f"class mismatch: clean C={clean_logits.shape[1]} noisy C={noisy_logits.shape[2]}"
    )


def _sample_variance(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[dim] < 2:
        return torch.zeros_like(x.select(dim, 0))
    return x.var(dim=dim, unbiased=True)


def tame_bias_variance(clean_logits: torch.Tensor, noisy_logits: torch.Tensor) -> dict:
    """TAME-style bias/variance summaries for stochastic logits.

    clean_logits: [B, C]
    noisy_logits: [B, K, C]
    """
    _assert_bkc_logits(clean_logits, noisy_logits)
    shift = noisy_logits.float() - clean_logits.float().unsqueeze(1)
    mu = shift.mean(dim=1)
    var = _sample_variance(shift, dim=1)
    return {
        "C_sum": float(mu.abs().max(dim=0).values.sum()),
        "V_sum": float(var.max(dim=0).values.sum()),
        "mean_abs_shift": float(shift.abs().mean()),
    }


def margin_cert_rate(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    labels: torch.Tensor,
    phi: float = 0.05,
) -> float:
    """Return the fraction of examples passing a one-sided margin certificate."""
    _assert_bkc_logits(clean_logits, noisy_logits)
    assert labels.dim() == 1, "labels must be [B]"
    assert labels.shape[0] == clean_logits.shape[0], (
        f"label batch mismatch: labels B={labels.shape[0]} clean B={clean_logits.shape[0]}"
    )
    if not (0.0 < float(phi) < 1.0):
        raise ValueError(f"phi must be in (0, 1), got {phi}")

    clean = clean_logits.float()
    noisy = noisy_logits.float()
    y = labels.to(device=clean.device, dtype=torch.long)
    rows = torch.arange(clean.shape[0], device=clean.device)

    true_clean = clean.gather(1, y[:, None]).squeeze(1)
    masked = clean.clone()
    masked[rows, y] = -1e30
    clean_margin = true_clean - masked.max(dim=1).values

    noisy_true = noisy.gather(2, y[:, None, None].expand(-1, noisy.shape[1], 1)).squeeze(2)
    noisy_other = noisy.clone()
    noisy_other[rows, :, y] = -1e30
    noisy_margin = noisy_true - noisy_other.max(dim=2).values

    shift = noisy_margin - clean_margin[:, None]
    mu = shift.mean(dim=1)
    var = _sample_variance(shift, dim=1)
    rhs = torch.sqrt(var * ((1.0 - float(phi)) / float(phi)))
    return float(((clean_margin + mu) > rhs).float().mean())


def diag_maha_ood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Diagonal Mahalanobis score along the last dimension."""
    return (((x.float() - mean.float()) ** 2) / var.float().clamp(min=1e-8)).sum(dim=-1)


def bootstrap_ci(
    values: torch.Tensor,
    stat_fn=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    vals = values.detach().float().reshape(-1)
    if vals.numel() == 0:
        raise ValueError("values must be non-empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if stat_fn is None:
        stat_fn = lambda x: x.mean()

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    stats = []
    n = vals.numel()
    for _ in range(int(n_boot)):
        idx = torch.randint(n, (n,), generator=gen)
        stat = stat_fn(vals[idx])
        if isinstance(stat, torch.Tensor):
            stat = float(stat.detach())
        stats.append(float(stat))
    q = torch.tensor(stats).quantile(torch.tensor([float(alpha) / 2.0, 1.0 - float(alpha) / 2.0]))
    return float(q[0]), float(q[1])
