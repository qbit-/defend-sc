"""Privacy-span positions shared by geometry and the SIPIT eval."""
from __future__ import annotations

from typing import Any

import torch


def positions_from_mask(
    privacy_mask: torch.Tensor,
    offsets: tuple[int, ...],
) -> list[int]:
    """Return absolute positions at fixed offsets into each private span.

    A position is kept when it lands on a private token for at least one
    row. Offset 0 is the first private token of that row.

    Args:
        privacy_mask: ``[B, T]`` mask, 1 on tokens the attack may score.
        offsets: Distances from each row's first private token.

    Returns:
        Sorted unique positions. Empty when no row has a private token.
    """
    found: set[int] = set()
    for row in range(privacy_mask.shape[0]):
        index = privacy_mask[row].nonzero(as_tuple=False).flatten()
        if index.numel() == 0:
            continue
        start = int(index[0])
        private = {int(item) for item in index.tolist()}
        for offset in offsets:
            position = start + int(offset)
            if position in private:
                found.add(position)
    return sorted(found)


def add_privacy_arguments(parser: Any) -> None:
    """Add the shared attack-position flags.

    Args:
        parser: ``argparse`` parser to extend.
    """
    parser.add_argument(
        "--positions", nargs="+", type=int, default=None,
        help=(
            "Privacy offsets of the attacked tokens. "
            "Omit to keep the task default."
        ),
    )
    parser.add_argument(
        "--offset_pattern", default="",
        help=(
            "Where --positions are applied. Empty applies "
            "them as absolute token indices. Otherwise "
            "offsets are counted after the last match."
        ),
    )


def configure_privacy(
    task: Any,
    positions: tuple[int, ...] | list[int] | None,
    offset_pattern: str,
) -> None:
    """Store CLI offsets on a task.

    An empty pattern leaves a mask-based task unchanged until
    ``positions`` is also set. A non-empty pattern anchors the
    offsets after the last match of that text.

    Args:
        task: Benchmark with privacy attributes.
        positions: Offsets, or ``None`` to keep the task default.
        offset_pattern: Anchor text. Empty selects absolute
            indices once ``positions`` is set.
    """
    if positions is not None:
        task.privacy_offsets = tuple(int(item) for item in positions)
        if offset_pattern == "":
            task.privacy_from_mask = False
            task.offset_pattern = ""
    if offset_pattern != "":
        task.offset_pattern = offset_pattern
        task.privacy_from_mask = False


def resolve_task_positions(
    task: Any,
    encoded: dict,
    tokenizer: Any = None,
) -> tuple[list[int], torch.Tensor | None]:
    """Return attack indices and an optional score mask.

    A mask-based task with an empty pattern uses its privacy span.
    Any other empty pattern treats the offsets as absolute indices.
    A pattern anchors offset 0 at the first token that extends
    past the last match.

    Args:
        task: Benchmark with privacy offsets.
        encoded: Batch with ids and masks.
        tokenizer: Required when ``offset_pattern`` is set.

    Returns:
        Absolute positions, and a score mask. ``None`` means the
        caller should keep the stored privacy mask.

    Raises:
        ValueError: If a pattern is set and no row contains it,
            or a pattern is set without a tokenizer.
    """
    pattern = str(getattr(task, "offset_pattern", "") or "")
    offsets = tuple(int(item) for item in task.privacy_offsets)
    if pattern:
        return _pattern_sites(encoded, offsets, pattern, tokenizer)
    if getattr(task, "privacy_from_mask", False):
        return positions_from_mask(encoded["privacy_mask"], offsets), None
    return _absolute_sites(encoded, offsets)


def _attention(encoded: dict) -> torch.Tensor:
    """Return the real-token mask for a batch.

    Args:
        encoded: Batch with an attention or privacy mask.

    Returns:
        Integer mask, ``[B, T]``.
    """
    if encoded.get("attention_mask") is not None:
        return encoded["attention_mask"].long()
    return encoded["privacy_mask"].long()


def _absolute_sites(
    encoded: dict, offsets: tuple[int, ...],
) -> tuple[list[int], torch.Tensor]:
    """Mark absolute offsets that lie inside the padded width.

    Token 0 is omitted. The SIPIT scorer skips it as well.

    Args:
        encoded: Batch with ``input_ids`` and an attention mask.
        offsets: Absolute token indices.

    Returns:
        Kept positions and a mask that is 1 on those real tokens.
    """
    attention = _attention(encoded)
    width = int(encoded["input_ids"].shape[1])
    chosen = [int(pos) for pos in offsets if 0 < int(pos) < width]
    score = torch.zeros(attention.shape, dtype=torch.long)
    for pos in chosen:
        score[:, pos] = (attention[:, pos] > 0).long()
    return chosen, score


def _pattern_sites(
    encoded: dict,
    offsets: tuple[int, ...],
    pattern: str,
    tokenizer: Any,
) -> tuple[list[int], torch.Tensor]:
    """Apply offsets after the last match of ``pattern``.

    Args:
        encoded: Batch with ``input_ids`` and an attention mask.
        offsets: Distances from the first token past the match.
        pattern: Anchor text. The last match in each row is used.
        tokenizer: Tokenizer used to recover the prompt text.

    Returns:
        Absolute positions and a mask that is 1 only there.

    Raises:
        ValueError: If ``tokenizer`` is missing or no row matches.
    """
    if tokenizer is None:
        raise ValueError("offset_pattern requires a tokenizer")
    ids = encoded["input_ids"]
    attention = _attention(encoded)
    score = torch.zeros(attention.shape, dtype=torch.long)
    found: set[int] = set()
    matched = False
    width = int(ids.shape[1])
    for row in range(ids.shape[0]):
        length = int((attention[row] > 0).sum().item())
        anchor = _anchor_after_pattern(
            tokenizer, ids[row, :length].tolist(), pattern,
        )
        if anchor is None:
            continue
        matched = True
        _mark_offsets(
            score, found, row, anchor, offsets, width, attention[row],
        )
    if not matched:
        raise ValueError(
            f"offset_pattern {pattern!r} matched no prompt",
        )
    return sorted(found), score


def _mark_offsets(
    score: torch.Tensor,
    found: set[int],
    row: int,
    anchor: int,
    offsets: tuple[int, ...],
    width: int,
    attention_row: torch.Tensor,
) -> None:
    """Record in-range offsets for one row.

    Args:
        score: ``[B, T]`` mask updated in place.
        found: Absolute positions seen so far.
        row: Batch index.
        anchor: Token index of offset 0.
        offsets: Distances from ``anchor``.
        width: Padded sequence length.
        attention_row: Real-token mask for this row.
    """
    for offset in offsets:
        position = anchor + int(offset)
        if position <= 0 or position >= width:
            continue
        if int(attention_row[position]) <= 0:
            continue
        found.add(position)
        score[row, position] = 1


def _decode_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode ids without cleaning up spaces.

    Args:
        tokenizer: Tokenizer with ``decode``.
        token_ids: Token ids.

    Returns:
        Decoded text.
    """
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids)


def _char_ends(tokenizer: Any, token_ids: list[int]) -> list[int]:
    """Return the exclusive character end of each prefix.

    Args:
        tokenizer: Tokenizer with ``decode``.
        token_ids: Token ids.

    Returns:
        One end index per token.
    """
    ends: list[int] = []
    for index in range(1, len(token_ids) + 1):
        text = _decode_prefix(tokenizer, token_ids[:index])
        ends.append(len(text))
    return ends


def _anchor_after_pattern(
    tokenizer: Any, token_ids: list[int], pattern: str,
) -> int | None:
    """Return the first token that extends past ``pattern``.

    Args:
        tokenizer: Tokenizer with ``decode``.
        token_ids: Real token ids for one prompt.
        pattern: Text to find. The last match wins.

    Returns:
        Token index, or ``None`` when the pattern is absent.
    """
    if not token_ids or not pattern:
        return None
    ends = _char_ends(tokenizer, token_ids)
    text = _decode_prefix(tokenizer, token_ids)
    start = text.rfind(pattern)
    if start < 0:
        return None
    pattern_end = start + len(pattern)
    for index, char_end in enumerate(ends):
        if char_end > pattern_end:
            return index
    return None
