from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def pad_embeddings_and_masks(
    embeddings: list[Tensor],
    masks: list[Tensor],
    target_len: int | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """Pad variable-length multi-vector embeddings and masks to a common seq length.

    When ``target_len`` is given, pad to that length; otherwise pad to the max
    across the provided lists.  Embeddings are zero-padded (masked out by the
    corresponding ``False`` entries in the mask during MaxSim scoring).
    """
    if target_len is None:
        target_len = max(e.size(1) for e in embeddings)
    embeddings = [
        F.pad(e, (0, 0, 0, target_len - e.size(1)))
        if e.size(1) < target_len
        else e
        for e in embeddings
    ]
    masks = [
        F.pad(m, (0, target_len - m.size(1)), value=False)
        if m.size(1) < target_len
        else m
        for m in masks
    ]
    return embeddings, masks
