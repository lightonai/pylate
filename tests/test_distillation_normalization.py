from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch

from pylate.evaluation import ColBERTDistillationEvaluator
from pylate.losses import Distillation
from pylate.scores._normalization import normalize_distillation_scores


class FixedEncoder(torch.nn.Module):
    skiplist = []
    do_query_expansion = False

    def forward(self, features):
        return {"token_embeddings": torch.ones(len(features["input_ids"]), 1, 2)}


def features():
    return [
        {
            "input_ids": torch.ones(n, 1, dtype=torch.long),
            "attention_mask": torch.ones(n, 1, dtype=torch.long),
        }
        for n in [1, 4]
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("values", [[2, 2, 2, 2], [0, 1, 2, 4]])
@pytest.mark.parametrize("normalize", [True, False])
def test_distillation_loss_and_gradients_are_finite(dtype, values, normalize):
    scores = torch.tensor([values], dtype=dtype, requires_grad=True)
    loss = Distillation(
        model=FixedEncoder(),
        score_metric=lambda *args, **kwargs: scores,
        normalize_scores=normalize,
    )(features(), torch.tensor([[2.0, 0.0, -1.0, 1.0]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_small_float32_range_retains_additive_epsilon():
    scores = torch.tensor([[1e-10, 2e-10, 4e-10, 8e-10]], dtype=torch.float32)
    expected = torch.tensor([[0.0, 1 / 107, 3 / 107, 7 / 107]])
    torch.testing.assert_close(normalize_distillation_scores(scores), expected)


def test_normalization_is_rowwise_and_preserves_integer_score_support():
    scores = torch.tensor([[1, 2, 4, 8], [4, 5, 7, 11]])
    expected = torch.tensor([[0.0, 1 / 7, 3 / 7, 1.0]]).expand(2, -1)
    torch.testing.assert_close(normalize_distillation_scores(scores), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("normalize", [True, False])
def test_distillation_evaluation_of_equal_scores_is_finite(
    monkeypatch, dtype, normalize
):
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [torch.ones(1, 2) for _ in texts]
    monkeypatch.setattr(
        "pylate.evaluation.colbert_distillation.colbert_kd_scores",
        lambda **kwargs: torch.full((1, 4), 2.0, dtype=dtype),
    )
    teacher = [2.0, 0.0, -1.0, 1.0]
    evaluator = ColBERTDistillationEvaluator(
        queries=["query"],
        documents=[["a", "b", "c", "d"]],
        scores=[teacher],
        normalize_scores=normalize,
        write_csv=False,
    )
    value = evaluator(model)["kl_divergence"]
    probabilities = [math.exp(x) / sum(math.exp(y) for y in teacher) for x in teacher]
    # log_softmax of equal low-precision scores is a rounded representation of -log(4).
    log_uniform = torch.tensor(-math.log(4), dtype=dtype).item()
    expected = sum(p * (math.log(p) - log_uniform) for p in probabilities)
    assert math.isfinite(value)
    assert value == pytest.approx(expected, abs=1e-6)


def test_tiny_float16_range_has_finite_backward():
    scores = torch.tensor(
        [[0.0, 0.0, 0.0, 6e-8]], dtype=torch.float16, requires_grad=True
    )
    loss = Distillation(
        model=FixedEncoder(), score_metric=lambda *args, **kwargs: scores
    )(features(), torch.tensor([[2.0, 0.0, -1.0, 1.0]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
