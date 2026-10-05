"""Greedy generation from a noisy prompt cut activation."""
from __future__ import annotations

import torch

from . import split_model as SM


def right_to_left_pad(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    pad_id: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Left-pad right-padded prompt rows so new tokens can append.

    Args:
        token_ids: Right-padded ids, ``[B, T]``.
        attention_mask: 1 on real prompt tokens.
        hidden: Cut activations aligned with ``token_ids``.
        pad_id: Token id written into the left pad.

    Returns:
        Left-padded ids, mask, hidden, start indices, and lengths.
    """
    lengths = attention_mask.sum(dim=1).long()
    batch, _time, hidden_size = hidden.shape
    width = max(int(lengths.max()), 1)
    ids = token_ids.new_full((batch, width), int(pad_id))
    mask = attention_mask.new_zeros(batch, width)
    packed = hidden.new_zeros(batch, width, hidden_size)
    starts = []
    for row in range(batch):
        length = int(lengths[row])
        start = width - length
        if length:
            real = attention_mask[row].bool()
            ids[row, start:] = token_ids[row, real]
            mask[row, start:] = 1
            packed[row, start:] = hidden[row, real]
        starts.append(start)
    start_tensor = torch.tensor(starts, dtype=torch.long)
    return ids, mask, packed, start_tensor, lengths


def _stop_ids(tokenizer) -> set[int]:
    """Return token ids that end a one-line answer.

    Args:
        tokenizer: Hugging Face tokenizer.

    Returns:
        EOS id, when set, plus ids produced by encoding a newline.
    """
    stops = set(tokenizer.encode("\n", add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is not None:
        stops.add(int(eos))
    return stops


def _append_token(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_id: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one column, padding rows that are already finished.

    Args:
        token_ids: Current ids.
        attention_mask: Current mask.
        next_id: Chosen token per row.
        active: True when the row should consume ``next_id``.

    Returns:
        Ids and mask grown by one column.
    """
    column = next_id[:, None]
    extra = active.to(dtype=attention_mask.dtype)[:, None]
    return (
        torch.cat([token_ids, column], dim=1),
        torch.cat([attention_mask, extra], dim=1),
    )


def _greedy_batch(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    k: int,
    max_new_tokens: int,
    pad_id: int,
) -> list[str]:
    """Greedy-decode one left-padded batch from spliced prompt activations.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer used to decode new tokens.
        token_ids: Right-padded prompt ids.
        attention_mask: Prompt mask.
        hidden: Prompt cut activations, same layout as ``token_ids``.
        k: Split depth.
        max_new_tokens: Maximum continuation length.
        pad_id: Pad token id.

    Returns:
        Decoded continuations, one string per row.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids, mask, packed, starts, lengths = right_to_left_pad(
        token_ids, attention_mask, hidden, pad_id,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    packed = packed.to(device=device, dtype=dtype)
    stops = _stop_ids(tokenizer)
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
    pieces: list[list[int]] = [[] for _ in range(ids.shape[0])]
    for _step in range(max_new_tokens):
        if bool(done.all()):
            break
        out = SM.split_run_splice(
            model, ids, k, packed, starts, lengths, mask,
        )
        chosen = out["logits"][:, -1, :].argmax(dim=-1)
        active = ~done
        for row in range(ids.shape[0]):
            if bool(done[row]):
                chosen[row] = pad_id
                continue
            token = int(chosen[row])
            text = tokenizer.decode([token], skip_special_tokens=False)
            if token in stops or "\n" in text:
                done[row] = True
                chosen[row] = pad_id
                continue
            pieces[row].append(token)
        ids, mask = _append_token(ids, mask, chosen, active & ~done)
    return [
        tokenizer.decode(row, skip_special_tokens=True) for row in pieces
    ]


@torch.no_grad()
def greedy_continuations(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    k: int,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    pad_id: int | None = None,
) -> list[str]:
    """Generate one-line continuations from prompt cut activations.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer for decoding and the newline stop.
        token_ids: Right-padded prompt ids.
        attention_mask: 1 on real prompt tokens.
        hidden: Prompt cut activations aligned with ``token_ids``.
        k: Split depth.
        max_new_tokens: Maximum new tokens per row.
        batch_size: Rows per forward.
        pad_id: Pad id. Defaults to the tokenizer pad or EOS id.

    Returns:
        One continuation string per prompt row.
    """
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
    texts: list[str] = []
    count = token_ids.shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        texts.extend(_greedy_batch(
            model,
            tokenizer,
            token_ids[start:stop],
            attention_mask[start:stop],
            hidden[start:stop],
            k,
            max_new_tokens,
            int(pad_id),
        ))
    return texts
