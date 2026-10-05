"""Privacy-span positions shared by geometry and the SIPIT eval."""
from __future__ import annotations

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
