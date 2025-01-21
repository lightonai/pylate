"""Tests the training loop."""

from __future__ import annotations

import os
import shutil

import pandas as pd
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from pylate import evaluation, losses, models, utils


def test_contrastive_training() -> None:
    """Test constrastive training."""
    if os.path.exists(path="tests/contrastive"):
        shutil.rmtree("tests/contrastive")

    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

    dataset = load_dataset("lightonai/lighton-ms-marco-mini", "triplet", split="train")

    splits = dataset.train_test_split(test_size=0.5)

    train_dataset, eval_dataset = splits["train"], splits["test"]

    train_loss = losses.Contrastive(model=model)

    dev_evaluation = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="tests/contrastive",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        learning_rate=3e-6,
        do_eval=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluation,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()

    model.save_pretrained("tests/contrastive/final")

    assert os.path.isdir("tests/contrastive")

    metrics = dev_evaluation(
        model=model,
        output_path="tests/contrastive/",
    )

    assert isinstance(metrics, dict)

    assert os.path.isfile(path="tests/contrastive/triplet_evaluation_results.csv")

    results = pd.read_csv(
        filepath_or_buffer="tests/contrastive/triplet_evaluation_results.csv"
    )

    assert "accuracy" in list(results.columns)

    if os.path.exists(path="tests/contrastive"):
        shutil.rmtree("tests/contrastive")
