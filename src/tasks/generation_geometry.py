"""Task subspace from teacher-forced gold-token logits."""
from __future__ import annotations

import torch

from src import models as M
from src import split_model as SM
from src.generate import right_to_left_pad


def subspace_from_responses(
    responses: torch.Tensor,
    probes: torch.Tensor,
    energy: float = 0.9,
    max_rank: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map a response matrix back to a cut-space subspace.

    Args:
        responses: ``[n_dirs, R]`` response of each probe.
        probes: Probe directions, ``[n_dirs, H]``.
        energy: Cumulative singular-value energy to keep.
        max_rank: Cap on the returned rank.

    Returns:
        ``U_T`` with shape ``[H, r]``, the kept singular values, and ``r``.
    """
    _left, singular, _right = torch.linalg.svd(
        responses.float(), full_matrices=False,
    )
    variance = singular.pow(2)
    cumulative = variance.cumsum(0) / variance.sum().clamp(min=1e-30)
    kept = singular.shape[0]
    if (cumulative >= energy).any():
        kept = int((cumulative >= energy).nonzero()[0].item()) + 1
    rank = max(1, min(kept, max_rank, int(singular.shape[0])))
    basis = probes.cpu().float().T @ _left[:, :rank]
    orthogonal, _rest = torch.linalg.qr(basis)
    return orthogonal, singular[:rank], rank


def _teacher_force_ids(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    gold_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    pad_id: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, int,
]:
    """Left-pad the prompt and append the gold continuation.

    Args:
        token_ids: Right-padded prompt ids.
        attention_mask: Prompt mask.
        hidden: Prompt cut activations.
        gold_ids: Right-padded gold token ids.
        gold_mask: 1 on real gold tokens.
        pad_id: Pad token id.

    Returns:
        Full ids, mask, left-padded prompt hidden, starts, lengths,
        and the left-padded prompt width.
    """
    ids, mask, packed, starts, lengths = right_to_left_pad(
        token_ids, attention_mask, hidden, pad_id,
    )
    gold_width = int(gold_ids.shape[1])
    batch, prompt_width = ids.shape
    total = prompt_width + gold_width
    full_ids = ids.new_full((batch, total), pad_id)
    full_mask = mask.new_zeros(batch, total)
    full_ids[:, :prompt_width] = ids
    full_mask[:, :prompt_width] = mask
    for row in range(batch):
        count = int(gold_mask[row].sum())
        if count == 0:
            continue
        full_ids[row, prompt_width:prompt_width + count] = gold_ids[row, :count]
        full_mask[row, prompt_width:prompt_width + count] = 1
    return full_ids, full_mask, packed, starts, lengths, prompt_width


def gold_logit_values(
    logits: torch.Tensor,
    prompt_width: int,
    gold_ids: torch.Tensor,
    gold_mask: torch.Tensor,
) -> torch.Tensor:
    """Pack the logit of each gold token into one vector.

    Left padding puts every prompt's last token at ``prompt_width - 1``.
    That position predicts the first gold token.

    Args:
        logits: ``[B, T, V]`` teacher-forced logits.
        prompt_width: Width of the left-padded prompt prefix.
        gold_ids: Gold token ids.
        gold_mask: 1 on real gold tokens.

    Returns:
        1-D tensor of gold-token logits in row-major order.
    """
    values = []
    for row in range(logits.shape[0]):
        count = int(gold_mask[row].sum())
        for index in range(count):
            token = int(gold_ids[row, index])
            position = prompt_width - 1 + index
            values.append(logits[row, position, token])
    if not values:
        raise ValueError("gold continuation is empty")
    return torch.stack(values)


def _chunk_response(
    model,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    gold_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    k: int,
    pad_id: int,
) -> torch.Tensor:
    """Teacher-force one chunk and return packed gold logits.

    Args:
        model: Causal language model.
        token_ids: Right-padded prompt ids for the chunk.
        attention_mask: Prompt mask.
        hidden: Prompt cut activations.
        gold_ids: Gold ids for the chunk.
        gold_mask: Gold mask for the chunk.
        k: Split depth.
        pad_id: Pad token id.

    Returns:
        Packed gold-token logits on CPU.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    full_ids, full_mask, packed, starts, lengths, width = _teacher_force_ids(
        token_ids.cpu(),
        attention_mask.cpu(),
        hidden.cpu(),
        gold_ids.cpu(),
        gold_mask.cpu(),
        pad_id,
    )
    out = SM.split_run_splice(
        model,
        full_ids.to(device),
        k,
        packed.to(device=device, dtype=dtype),
        starts,
        lengths,
        full_mask.to(device),
    )
    return gold_logit_values(
        out["logits"], width, gold_ids.to(device), gold_mask.to(device),
    ).cpu()


def _responses_for_hidden(
    model,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    gold_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    k: int,
    pad_id: int,
    batch_size: int,
) -> torch.Tensor:
    """Pack gold logits for a batch, in row order.

    Args:
        model: Causal language model.
        token_ids: Right-padded prompt ids.
        attention_mask: Prompt mask.
        hidden: Prompt cut activations.
        gold_ids: Gold ids.
        gold_mask: Gold mask.
        k: Split depth.
        pad_id: Pad token id.
        batch_size: Rows per forward.

    Returns:
        Concatenated gold-token logits.
    """
    parts = []
    count = token_ids.shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        parts.append(_chunk_response(
            model,
            token_ids[start:stop],
            attention_mask[start:stop],
            hidden[start:stop],
            gold_ids[start:stop],
            gold_mask[start:stop],
            k,
            pad_id,
        ))
    return torch.cat(parts, dim=0)


def _prompt_delta(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Add ``tau * direction`` on every real prompt position.

    Args:
        hidden: Prompt cut activations.
        attention_mask: 1 on real prompt tokens.
        direction: Unit probe, shape ``[H]``.
        tau: Finite-difference step.

    Returns:
        Perturbed activations with the same shape as ``hidden``.
    """
    step = (tau * direction).to(device=hidden.device, dtype=hidden.dtype)
    scale = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + step.view(1, 1, -1) * scale.unsqueeze(-1)


@torch.no_grad()
def gold_logit_jacobian(
    model,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    clean_hidden: torch.Tensor,
    gold_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    probes: torch.Tensor,
    k: int,
    tau: float = 1e-2,
    energy: float = 0.9,
    max_rank: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Estimate ``U_T`` from gold-token logit changes.

    Args:
        model: Causal language model.
        token_ids: Right-padded prompt ids.
        attention_mask: Prompt mask.
        clean_hidden: Clean prompt cut activations.
        gold_ids: Teacher-forced continuation ids.
        gold_mask: 1 on real continuation tokens.
        probes: Probe directions, ``[n_dirs, H]``.
        k: Split depth.
        tau: Finite-difference step.
        energy: Singular-value energy to keep.
        max_rank: Cap on the subspace rank.

    Returns:
        ``U_T``, singular values, and the kept rank.
    """
    missing = attention_mask == 0
    if bool(missing.any()):
        pad_id = int(token_ids[missing][0])
    else:
        pad_id = 0
    batch_size = M.forward_batch_size(model)
    clean = _responses_for_hidden(
        model, token_ids, attention_mask, clean_hidden,
        gold_ids, gold_mask, k, pad_id, batch_size,
    )
    rows = []
    for direction in probes:
        vector = direction.to(clean_hidden.device, clean_hidden.dtype)
        vector = vector / vector.norm().clamp(min=1e-12)
        perturbed = _prompt_delta(
            clean_hidden, attention_mask, vector, tau,
        )
        shifted = _responses_for_hidden(
            model, token_ids, attention_mask, perturbed,
            gold_ids, gold_mask, k, pad_id, batch_size,
        )
        rows.append((shifted - clean).cpu())
    response = torch.stack(rows, dim=0)
    return subspace_from_responses(
        response, probes, energy=energy, max_rank=max_rank,
    )
