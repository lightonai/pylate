"""Triplet/distillation evaluators should record epoch and step on the model card."""

from __future__ import annotations

import torch

from pylate import evaluation


class _DummyEncoder:
    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        return torch.ones(len(sentences), 3, 4)


def test_triplet_evaluator_passes_epoch_and_step_to_model_card(monkeypatch) -> None:
    captured: dict = {}

    def fake_store(self, model, metrics, epoch=0, step=0):
        captured["epoch"] = epoch
        captured["step"] = step
        captured["metrics"] = dict(metrics)

    monkeypatch.setattr(
        evaluation.ColBERTTripletEvaluator,
        "store_metrics_in_model_card_data",
        fake_store,
    )

    evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=["fruits are healthy."],
        positives=["fruits are good for health."],
        negatives=["chips are junk food."],
        name="dev",
        write_csv=False,
    )
    evaluator(model=_DummyEncoder(), epoch=2, steps=17)

    assert captured["epoch"] == 2
    assert captured["step"] == 17
    assert "dev_accuracy" in captured["metrics"]


def test_distillation_evaluator_passes_epoch_and_step_to_model_card(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_store(self, model, metrics, epoch=0, step=0):
        captured["epoch"] = epoch
        captured["step"] = step
        captured["metrics"] = dict(metrics)

    monkeypatch.setattr(
        evaluation.ColBERTDistillationEvaluator,
        "store_metrics_in_model_card_data",
        fake_store,
    )

    evaluator = evaluation.ColBERTDistillationEvaluator(
        queries=["query A"],
        documents=[["document A", "document B"]],
        scores=[[0.9, 0.1]],
        write_csv=False,
        normalize_scores=False,
    )
    evaluator(model=_DummyEncoder(), epoch=3, steps=9)

    assert captured["epoch"] == 3
    assert captured["step"] == 9
    assert "kl_divergence" in captured["metrics"]
