"""SST-2-specific geometry: T_k uses the 2-class label logit as the downstream map."""
from __future__ import annotations
import torch
from . import models as M
from . import split_model as SM


@torch.no_grad()
def label_logit_jacobian_subspace(model, ids, mask, ans_pos, label_token_ids, k,
                                   probe_dirs, tau=1e-2, energy=0.9, max_rank=64):
    """Estimate U_T as the leading subspace of cut-space directions whose
    perturbation produces large change in the *answer-position label logits*.

    For each probe direction u, perturb a_k at the *answer position* of every
    example by tau*u, run server tail, measure response in label logits [B, 2].
    Stack responses [n_dirs, B, 2] -> SVD over (B*2) dim to extract leading
    subspace in cut-space (n_dirs side).
    """
    out = SM.split_run(model, ids, k=k, attention_mask=mask)
    a_k = out["a_k"]
    label_ids_t = torch.tensor(label_token_ids, device=a_k.device)
    pos = ans_pos.view(-1, 1, 1).expand(-1, 1, out["logits"].shape[-1])
    last_clean = out["logits"].gather(1, pos).squeeze(1)
    label_logits_clean = last_clean.index_select(-1, label_ids_t)        # [B, 2]
    B, T, H = a_k.shape
    n_dirs = probe_dirs.shape[0]
    R = []
    for u in probe_dirs:
        u = u.to(a_k.device, a_k.dtype)
        u = u / u.norm().clamp(min=1e-12)
        delta = torch.zeros_like(a_k)
        # perturb at each row's answer position
        idx = ans_pos.view(B, 1, 1).expand(B, 1, H)
        delta.scatter_(1, idx, (tau * u).expand(B, 1, H))
        out_p = SM.split_run(model, ids, k=k, hidden_override=a_k + delta,
                             attention_mask=mask)
        last_p = out_p["logits"].gather(1, pos).squeeze(1)
        label_p = last_p.index_select(-1, label_ids_t)                    # [B, 2]
        resp = (label_p - label_logits_clean).reshape(-1)                # flatten over (B, 2)
        R.append(resp.cpu())
    R = torch.stack(R, dim=0).float()                                    # [n_dirs, B*2]
    U_left, S, Vh = torch.linalg.svd(R, full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum().clamp(min=1e-30)
    r = int((cum >= energy).nonzero()[0].item()) + 1 if (cum >= energy).any() else len(S)
    r = max(1, min(r, max_rank, len(S)))
    # leading subspace in cut-space = leading left singular vectors of R
    # R is [n_dirs, B*2]; we want the cut-space basis of "directions of strong
    # response", which corresponds to the rows of R via U_left projecting probe
    # directions. The leading cut-space subspace is U_T = probes^T @ U_left[:, :r].
    probes_cpu = probe_dirs.cpu().float()
    U_T = probes_cpu.T @ U_left[:, :r]                                   # [H, r]
    Q, _ = torch.linalg.qr(U_T)
    return Q, S[:r], r
