"""Candidate-token cloud generation for Phase 2 subspace alignment.

For a (prompt prefix, position t), pick V' candidate tokens from:
  - top-k next-token logits (model's own predictions)
  - random vocab controls
Compute cut-layer hidden state h_t(prefix || v) for each v in V', then SVD to
extract the local SIPIT-discriminative subspace S(pi, t).
"""
from __future__ import annotations
import torch
from . import models as M
from . import split_model as SM


@torch.no_grad()
def topk_candidates(model, input_ids, position: int, k_top: int = 200,
                     attention_mask=None) -> torch.Tensor:
    """Return [batch, k_top] candidate token IDs from model's next-token logits."""
    out = model(input_ids=input_ids[:, : position + 1], attention_mask=None)
    logits = out.logits[:, -1, :]   # [B, V]
    return logits.topk(k_top, dim=-1).indices  # [B, k_top]


@torch.no_grad()
def random_candidates(vocab_size: int, batch: int, k_top: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, k_top), generator=g)


@torch.no_grad()
def candidate_cloud(model, prefix_ids, position: int, candidate_ids,
                    k_split: int, chunk_size: int = 256) -> torch.Tensor:
    """Compute cut-layer hidden state for each candidate appended after prefix.

    prefix_ids: [B, P] (we use the first `position` tokens as prefix)
    candidate_ids: [B, V'] candidate token ids appended at position t.
    Returns: [B, V', hidden]

    Chunked along the (B*V) axis to control peak memory; the relevant attention
    cost is O((P+1)^2 * batch_chunk * H) so very long sequences should use a
    smaller chunk.
    """
    device = prefix_ids.device
    B, V = candidate_ids.shape
    prefix = prefix_ids[:, : position]
    P = prefix.shape[1]
    pref_rep = prefix.unsqueeze(1).expand(-1, V, -1).reshape(B * V, P)
    cand = candidate_ids.reshape(B * V, 1)
    seq = torch.cat([pref_rep, cand], dim=1)
    H_dim = None
    chunks = []
    for i in range(0, seq.shape[0], chunk_size):
        s = seq[i: i + chunk_size]
        a_k = SM.capture_a_k(model, s, k=k_split)      # [chunk, P+1, hidden]
        chunks.append(a_k[:, -1, :].cpu())             # bring to CPU to avoid VRAM accumulation
        if H_dim is None: H_dim = chunks[-1].shape[-1]
        del a_k
    h_t = torch.cat(chunks, dim=0)
    return h_t.reshape(B, V, H_dim)


def local_subspace(cloud: torch.Tensor, energy: float = 0.9, max_rank: int = 64
                   ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Center cloud and compute SVD, retain top components covering `energy` variance.

    cloud: [V, hidden]
    returns: U[hidden, r], singular_values[r], r
    """
    X = cloud - cloud.mean(dim=0, keepdim=True)        # [V, H]
    if X.shape[0] < 2:
        return torch.zeros(X.shape[1], 0), torch.zeros(0), 0
    U_left, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum().clamp(min=1e-30)
    r = int((cum >= energy).nonzero()[0].item()) + 1 if (cum >= energy).any() else len(S)
    r = max(1, min(r, max_rank, len(S)))
    return Vh[:r].T.contiguous(), S[:r].contiguous(), r
