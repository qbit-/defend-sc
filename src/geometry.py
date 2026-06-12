"""Privacy / utility geometry estimation: G_priv, G_util,k, principal angles, mass."""
from __future__ import annotations
import torch
from . import models as M
from . import split_model as SM


def covariance_outer(D_dirs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """G_priv = E[d d^T / ||d||^2] estimated from a set of difference vectors.

    D_dirs: [N, hidden] candidate-direction samples (e.g. h(prefix||v) - h(prefix||v')).
    """
    if normalize:
        n = D_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        Dn = D_dirs / n
    else:
        Dn = D_dirs
    return Dn.T @ Dn / max(1, Dn.shape[0])


def symmetrize_psd(A: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    """Return a symmetric floating-point copy, optionally with diagonal ridge."""
    out = A if A.dtype.is_floating_point else A.float()
    out = (out + out.T) / 2
    if ridge > 0:
        eye = torch.eye(out.shape[0], dtype=out.dtype, device=out.device)
        out = out + float(ridge) * eye
    return out


def lowrank_floor_maha(
    d: torch.Tensor,
    sigma0: float,
    U: torch.Tensor | None,
    lam: torch.Tensor | None,
) -> torch.Tensor:
    """Mahalanobis distance for Sigma = sigma0^2 I + U diag(lam) U.T.

    This is the Woodbury-form inverse for an orthonormal low-rank factor U.
    """
    d = d if d.dtype.is_floating_point else d.float()
    if float(sigma0) == 0:
        if U is None or lam is None or U.numel() == 0:
            return torch.zeros(d.shape[:-1], dtype=d.dtype, device=d.device)
        U = U.to(device=d.device, dtype=d.dtype)
        lam = lam.to(device=d.device, dtype=d.dtype).clamp(min=1e-12)
        proj = d @ U
        return (proj * proj / lam).sum(dim=-1)

    s2 = max(float(sigma0) ** 2, 1e-30)
    base = (d * d).sum(dim=-1) / s2
    if U is None or lam is None or U.numel() == 0:
        return base
    U = U.to(device=d.device, dtype=d.dtype)
    lam = lam.to(device=d.device, dtype=d.dtype).clamp(min=0)
    proj = d @ U
    correction = lam / (s2 * (s2 + lam).clamp(min=1e-30))
    return base - (proj * proj * correction).sum(dim=-1)


def repeated_separation(delta2: torch.Tensor, repeats=(1, 4, 16)) -> dict[int, torch.Tensor]:
    """Scale one-release squared separation by independent release count."""
    return {int(m): int(m) * delta2 for m in repeats}


def repeated_separation_summary(
    delta2: torch.Tensor,
    repeats=(1, 4, 16),
) -> dict[int, dict[str, float]]:
    """Small JSON-friendly diagnostics for repeated edge separation."""
    scaled = repeated_separation(delta2.float(), repeats=repeats)
    out = {}
    for m, vals in scaled.items():
        out[m] = {
            "p50": float(torch.quantile(vals, 0.50).cpu()),
            "p95": float(torch.quantile(vals, 0.95).cpu()),
            "max": float(vals.max().cpu()),
        }
    return out


def candidate_edge_directions(
    cloud: torch.Tensor,
    truth_index: torch.Tensor,
    max_edges_per_prompt: int = 16,
) -> torch.Tensor:
    """Build candidate-minus-truth SIPIT edge directions from a candidate cloud.

    cloud: [B, C, hidden], truth_index: [B].
    """
    dirs = []
    for b in range(cloud.shape[0]):
        true_idx = int(truth_index[b])
        true_vec = cloud[b, true_idx]
        diff = cloud[b] - true_vec.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        order = torch.argsort(dist2)
        picked = [int(i) for i in order.tolist() if int(i) != true_idx]
        for i in picked[:max_edges_per_prompt]:
            dirs.append(diff[i].detach().float().cpu())
    if not dirs:
        return torch.zeros(0, cloud.shape[-1])
    return torch.stack(dirs, dim=0)


def sipit_matrix_from_edges(edge_dirs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Assemble a PSD SIPIT edge sensitivity matrix."""
    if edge_dirs.numel() == 0:
        raise ValueError("edge_dirs must contain at least one SIPIT edge")
    eye = torch.eye(edge_dirs.shape[-1], dtype=edge_dirs.dtype, device=edge_dirs.device)
    return covariance_outer(edge_dirs, normalize=True) + float(eps) * eye


def generalized_privacy_directions(
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve top generalized directions B q = lambda (A + ridge I) q.

    Q_gen columns satisfy the generalized eigenproblem. Q_cov is a
    Euclidean-orthonormal basis spanning the same returned subspace for sampling.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    A_reg = symmetrize_psd(A, ridge=ridge)
    B = symmetrize_psd(B).to(device=A_reg.device, dtype=A_reg.dtype)
    max_rank = min(int(rank), A_reg.shape[0])
    L = torch.linalg.cholesky(A_reg)
    eye = torch.eye(A_reg.shape[0], dtype=A_reg.dtype, device=A_reg.device)
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
    C = symmetrize_psd(L_inv @ B @ L_inv.T)
    eigvals, eigvecs = torch.linalg.eigh(C)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    Q_gen = L_inv.T @ eigvecs[:, :max_rank]
    Q_cov, _ = torch.linalg.qr(Q_gen)
    return Q_cov[:, :max_rank].contiguous(), Q_gen.contiguous(), eigvals[:max_rank].contiguous()


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


@torch.no_grad()
def finite_diff_jacobian_subspace(model, ids, k: int, candidate_dirs: torch.Tensor,
                                  attention_mask=None, tau: float = 1e-2,
                                  energy: float = 0.9, max_rank: int = 64
                                  ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Estimate U_T (task subspace) by finite differences of D = server output.

    For each candidate direction u, perturb a_k by tau*u and accumulate
    G_util ~ sum (D(a + tau u) - D(a)) (D(a + tau u) - D(a))^T projected to cut space.

    Concretely: server output is in the same hidden space (residual stream).
    We treat the *response in server output* as our utility-relevant signal, and
    its column-space gives us U_T.

    candidate_dirs: [n_dirs, hidden] orthonormal-ish probe directions.
    Returns U_T: [hidden, r].
    """
    out_clean = SM.split_run(model, ids, k=k, attention_mask=attention_mask)
    a_k = out_clean["a_k"]
    server_clean = out_clean["server_out"]            # [B, T, hidden]

    # Stack response vectors at each (sample, position): we project each direction's
    # response onto its own perturbation -> we want subspace of significant responses.
    H = a_k.shape[-1]
    n_dirs = candidate_dirs.shape[0]
    # responses: list of vectors that change in cut space *as a function of u*.
    # Use each token-position's last-hidden response, average a few positions to
    # control variance.
    pos = a_k.shape[1] // 2                            # mid sequence position
    R = []
    for u in candidate_dirs:
        u = u.to(a_k.device, a_k.dtype) / u.norm().clamp(min=1e-12)
        delta = torch.zeros_like(a_k)
        delta[:, pos, :] = tau * u
        out_p = SM.split_run(model, ids, k=k, hidden_override=a_k + delta,
                             attention_mask=attention_mask)
        # response vector for this direction at the same position, averaged over batch
        resp = (out_p["server_out"][:, pos, :] - server_clean[:, pos, :]).mean(dim=0)
        R.append(resp)
    R = torch.stack(R, dim=0).float()                 # [n_dirs, hidden]
    # Use SVD of R to get the leading subspace of utility responses
    U_left, S, Vh = torch.linalg.svd(R, full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum().clamp(min=1e-30)
    r = int((cum >= energy).nonzero()[0].item()) + 1 if (cum >= energy).any() else len(S)
    r = max(1, min(r, max_rank, len(S)))
    U_T = Vh[:r].T.contiguous()
    return U_T, S[:r], r
