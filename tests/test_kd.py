"""Tests the training loop."""

import os
import shutil

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from giga_cherche import losses, models, utils


def test_kd_training() -> None:
    """Test knowledge distillation training."""
    if os.path.exists(path="tests/kd"):
        shutil.rmtree("tests/kd")

    train = load_dataset(
        path="lightonai/lighton-ms-marco-mini",
        name="train",
    )

    queries = load_dataset(
        path="lightonai/lighton-ms-marco-mini",
        name="queries",
    )

    documents = load_dataset(
        path="lightonai/lighton-ms-marco-mini",
        name="documents",
    )

    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

    args = SentenceTransformerTrainingArguments(
        output_dir="tests/kd",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        fp16=False,
        bf16=False,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        learning_rate=3e-6,
    )

    train_loss = losses.Distillation(model=model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()

    assert os.path.isdir("tests/kd")

    if os.path.exists(path="tests/kd"):
        shutil.rmtree("tests/kd")
