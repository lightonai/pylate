"""The min-max normalisation guard must survive reduced precision.

`Distillation` normalises its scores with
``(scores - min) / (max - min + epsilon)``. When a candidate group's scores are
equal the denominator is carried entirely by ``epsilon``, so ``epsilon`` has to be
representable in the dtype the scores arrive in. ``1e-8`` is not: in float16 it
rounds to zero.

The scores are injected through ``score_metric`` so the test exercises the
normalisation rather than any particular model's score distribution.
"""

from __future__ import annotations

import pytest
import torch

from pylate import losses, models

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def model() -> models.ColBERT:
    colbert = models.ColBERT(model_name_or_path=MODEL, device="cpu")
    colbert.train()
    return colbert


def _degenerate_scores(dtype: torch.dtype):
    """A group whose candidates all score the same, so max == min."""

    def score_metric(queries_embeddings, documents_embeddings, **kwargs):
        scores = torch.full((1, 4), 3.5, dtype=dtype)
        # keep the graph alive so gradients still flow to the model
        return scores + 0.0 * queries_embeddings.to(dtype).sum()

    return score_metric


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_normalisation_is_finite(model: models.ColBERT, dtype: torch.dtype) -> None:
    distillation = losses.Distillation(
        model=model, score_metric=_degenerate_scores(dtype)
    )
    features = [
        model.tokenize(["what causes programmed cell death"], is_query=True),
        model.tokenize(["apoptosis is programmed cell death"] * 4, is_query=False, pad=True),
    ]
    labels = torch.tensor([[0.4, 0.3, 0.2, 0.1]])

    for parameter in model.parameters():
        parameter.grad = None

    loss = distillation(sentence_features=features, labels=labels)
    assert torch.isfinite(loss), f"loss is {loss} with {dtype} scores"

    loss.backward()
    non_finite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not non_finite, f"{len(non_finite)} parameters have non-finite gradients"
