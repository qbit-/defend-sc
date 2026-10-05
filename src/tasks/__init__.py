"""Benchmark registry for the privacy/utility pipeline."""
from __future__ import annotations


def get_task(name: str):
    """Return the benchmark implementation for ``name``.

    Args:
        name: Benchmark id, ``sst2`` or ``word_sorting``.

    Returns:
        Task object with load, encode, and geometry hooks.

    Raises:
        ValueError: If ``name`` is not a known benchmark.
    """
    if name == "sst2":
        from .sst2 import SST2Task

        return SST2Task()
    if name == "word_sorting":
        from .word_sorting import WordSortingTask

        return WordSortingTask()
    raise ValueError(
        f"unknown task {name!r}; expected 'sst2' or 'word_sorting'"
    )
