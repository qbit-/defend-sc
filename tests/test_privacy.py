"""Shared privacy offsets for every benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks import get_task
from src.tasks.privacy import configure_privacy, resolve_task_positions
from src.tasks.privacy import positions_from_mask


class _CharTokenizer:
    """Map each code point to one token id."""

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        """Join character tokens.

        Args:
            token_ids: Code points.
            **_kwargs: Ignored tokenizer options.

        Returns:
            The decoded string.
        """
        return "".join(chr(int(item)) for item in token_ids)


def _ids(text: str) -> torch.Tensor:
    """Encode one string as a single batch row.

    Args:
        text: Prompt text, one character per token.

    Returns:
        ``input_ids`` of shape ``[1, T]``.
    """
    return torch.tensor([[ord(char) for char in text]])


def _encoded(text: str, privacy: list[int] | None = None) -> dict:
    """Build a one-row batch with a full attention mask.

    Args:
        text: Prompt text.
        privacy: Optional privacy bits. Defaults to all ones.

    Returns:
        Encoded batch dict.
    """
    ids = _ids(text)
    attention = torch.ones_like(ids)
    if privacy is None:
        mask = attention.clone()
    else:
        mask = torch.tensor([privacy])
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "privacy_mask": mask,
    }


def test_empty_pattern_uses_absolute_indices() -> None:
    """Offsets are token indices when no pattern is set."""
    task = get_task("sst2")
    configure_privacy(task, [2, 5, 8, 30], "")
    encoded = _encoded("x" * 12)
    positions, score = resolve_task_positions(task, encoded)
    assert positions == [2, 5, 8]
    assert int(score[0, 2]) == 1
    assert int(score[0, 1]) == 0
    assert int(score.sum()) == 3


def test_absolute_positions_ignore_the_privacy_span() -> None:
    """A position list is not measured from the private span."""
    task = get_task("word_sorting")
    configure_privacy(task, [1, 4], "")
    encoded = _encoded("abcdefghij", privacy=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    positions, score = resolve_task_positions(task, encoded)
    assert positions == [1, 4]
    assert int(score[0, 1]) == 1
    assert int(score[0, 3]) == 0


def test_pattern_uses_the_last_match() -> None:
    """Offsets start at the first token past the last match."""
    task = get_task("sst2")
    configure_privacy(task, [0, 1, 3], "List: ")
    encoded = _encoded("List: abList: xy")
    positions, score = resolve_task_positions(
        task, encoded, _CharTokenizer(),
    )
    text = "List: abList: xy"
    anchor = text.rfind("List: ") + len("List: ")
    assert positions == [anchor, anchor + 1]
    assert int(score[0, anchor]) == 1
    assert int(score[0, anchor + 1]) == 1
    assert int(score.sum()) == 2


def test_pattern_is_per_row() -> None:
    """Each row anchors its own copy of the pattern."""
    task = get_task("word_sorting")
    configure_privacy(task, [0], "List: ")
    first = "List: ab"
    second = "zzList: q"
    ids = torch.zeros(2, len(second), dtype=torch.long)
    attention = torch.zeros(2, len(second), dtype=torch.long)
    ids[0, :len(first)] = torch.tensor([ord(char) for char in first])
    ids[1] = torch.tensor([ord(char) for char in second])
    attention[0, :len(first)] = 1
    attention[1] = 1
    encoded = {
        "input_ids": ids,
        "attention_mask": attention,
        "privacy_mask": torch.zeros_like(attention),
    }
    positions, score = resolve_task_positions(
        task, encoded, _CharTokenizer(),
    )
    assert positions == [6, 8]
    assert int(score[0, 6]) == 1
    assert int(score[1, 8]) == 1
    assert int(score[0, 8]) == 0


def test_missing_pattern_raises() -> None:
    """A pattern that never occurs is an error."""
    task = get_task("sst2")
    configure_privacy(task, [0], "List: ")
    with pytest.raises(ValueError, match="matched no prompt"):
        resolve_task_positions(task, _encoded("no marker"), _CharTokenizer())


def test_word_sorting_default_stays_inside_the_span() -> None:
    """Unset flags keep offsets inside the private word list."""
    task = get_task("word_sorting")
    mask = torch.zeros(1, 8)
    mask[0, 3:8] = 1
    encoded = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "privacy_mask": mask,
    }
    positions, score = resolve_task_positions(task, encoded)
    assert score is None
    assert positions == positions_from_mask(
        mask, task.privacy_offsets,
    )
    assert positions == [3, 4, 5, 6]


def test_sst2_default_is_absolute() -> None:
    """SST-2 drops default offsets that do not fit the batch."""
    task = get_task("sst2")
    encoded = _encoded("x" * 12)
    positions, score = resolve_task_positions(task, encoded)
    assert positions == [2, 5, 10]
    assert score is not None
    assert int(score[0, 10]) == 1
