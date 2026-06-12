from __future__ import annotations
import torch
from . import models as M
from . import split_model as SM


def principal_angles(U_S: torch.Tensor, U_T: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between two column-orthonormal subspaces."""
    Q1, _ = torch.linalg.qr(U_S.float().cpu())
    Q2, _ = torch.linalg.qr(U_T.float().cpu())
    M = Q1.T @ Q2
    s = torch.linalg.svdvals(M).clamp(-1, 1)
    return torch.arccos(s)


def mass_T_perp_of_S(U_S: torch.Tensor, U_T: torch.Tensor) -> float:
    """Fraction of S mass that lies *outside* T's span, i.e. 1 - ||U_T^T U_S||_F^2 / r_S."""
    Q1, _ = torch.linalg.qr(U_S.float().cpu())
    Q2, _ = torch.linalg.qr(U_T.float().cpu())
    fro = (Q2.T @ Q1).pow(2).sum().item()
    r = Q1.shape[1]
    return 1.0 - fro / max(1, r)
