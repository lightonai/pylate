from __future__ import annotations

import torch


def normalize_distillation_scores(scores: torch.Tensor) -> torch.Tensor:
    """Min-max normalize rows with a denominator representable in the score dtype."""
    max_scores = scores.max(dim=1, keepdim=True).values
    min_scores = scores.min(dim=1, keepdim=True).values
    # Preserve the existing epsilon where possible; fp16 also needs a safe backward.
    denominator = max_scores - min_scores + 1e-8
    denominator = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    return (scores - min_scores) / denominator
