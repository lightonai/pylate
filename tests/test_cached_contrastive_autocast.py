"""CachedContrastive must recompute under the surrounding autocast context."""

from __future__ import annotations

import pytest
import torch

from pylate import losses, models
from pylate.losses.cached_contrastive import _capture_autocast_kwargs


def test_capture_autocast_kwargs_cpu_bf16() -> None:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        kwargs = _capture_autocast_kwargs()
    assert kwargs["enabled"] is True
    assert kwargs["device_type"] == "cpu"
    assert kwargs["dtype"] == torch.bfloat16

    kwargs = _capture_autocast_kwargs()
    assert kwargs["enabled"] is False


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is not supported")
def test_cached_contrastive_survives_surrounding_autocast() -> None:
    torch.manual_seed(0)
    model = models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    model.eval()
    sentence_features = [
        model.tokenize(
            ["fruits are healthy.", "chips are not healthy."], is_query=True
        ),
        model.tokenize(
            ["fruits are good for health.", "chips are junk food."],
            is_query=False,
            pad=True,
        ),
        model.tokenize(
            ["the sky is blue today.", "cars need fuel to run."],
            is_query=False,
            pad=True,
        ),
    ]
    loss_fn = losses.CachedContrastive(model=model, mini_batch_size=1)
    for parameter in model.parameters():
        parameter.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(sentence_features=sentence_features)
    loss.backward()
    assert isinstance(loss.item(), float)
