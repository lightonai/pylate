"""Script to reproduce the training of Agent-ModernColBERT, a fine-tuning of GTE-ModernColBERT on the AgentIR-data dataset.

Run with defaults via `accelerate launch agent_modern_colbert.py`, or override
any training param on the CLI, e.g. `--learning_rate 5e-6 --batch_size 64`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from typing import Callable

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models

# AgentIR's official query-side instruction prefix. Matches train.sh in the
# AgentIR repo, where bash double-quoted "\n" is a literal backslash-n (2
# chars), NOT a real newline. Tevatron receives that literal as-is and
# concatenates it to the query, so we mirror it byte-for-byte here.
INSTRUCT_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating the user's reasoning"
    "\\nQuery:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Agent-ModernColBERT on AgentIR-data."
    )
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="lightonai/GTE-ModernColBERT")
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--data_cache_dir", type=str, default="./data_cache/")
    return parser.parse_args()


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT model that samples a fixed number of negatives per batch."""

    def __init__(
        self,
        tokenize_fn: Callable,
        num_negatives: int = 7,
        valid_label_columns: list[str] | None = None,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        if valid_label_columns is None:
            valid_label_columns = ["label", "scores"]
        self.valid_label_columns = valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())
        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]
        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
            columns_to_process = other_columns + sampled_negatives
        else:
            columns_to_process = columns
        for column in columns_to_process:
            is_query = "query" in column or "anchor" in column
            texts = [row[column] for row in features]
            if isinstance(texts[0], list):
                texts = list(itertools.chain(*texts))
            tokenized = self.tokenize_fn(texts, is_query=is_query, pad=True)
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value
        return batch


def _convert_negatives_to_columns(example, max_negatives):
    """Flatten passages into negative_0..N columns and prepend the Instruct prefix to the query."""
    negatives = example["negative_passages"]
    for i in range(max_negatives):
        if i < len(negatives):
            example[f"negative_{i}"] = negatives[i]["text"]
        else:
            example[f"negative_{i}"] = ""
    example["positive"] = example["positive_passages"][0]["text"]
    example["query"] = INSTRUCT_PREFIX + example["query"]
    return example


def load_train_dataset(data_cache_dir: str) -> Dataset:
    os.makedirs(data_cache_dir, exist_ok=True)

    try:
        dataset = Dataset.load_from_disk(data_cache_dir)
        print(f"Loaded cached AgentIR (instruct) dataset from {data_cache_dir}.")
        return dataset
    except FileNotFoundError:
        pass

    print("Loading Tevatron/AgentIR-data dataset...")
    raw = load_dataset("Tevatron/AgentIR-data", split="train", num_proc=45)

    min_negatives = min(len(neg) for neg in raw["negative_passages"])
    max_negatives = min(min_negatives, 50)
    print(f"Min negatives across dataset: {min_negatives}, using {max_negatives}")

    print(
        "Converting passages to text columns and prepending Instruct prefix to queries..."
    )
    raw = raw.map(
        lambda x: _convert_negatives_to_columns(x, max_negatives),
        remove_columns=["query_id", "positive_passages", "negative_passages"],
        num_proc=11,
        desc="Converting to text",
    )

    neg_cols = sorted(
        [c for c in raw.column_names if c.startswith("negative_")],
        key=lambda c: int(c.split("_")[1]),
    )
    raw = raw.select_columns(["query", "positive"] + neg_cols)

    raw.save_to_disk(data_cache_dir)
    print(f"Saved processed dataset to {data_cache_dir}")
    return raw


def main():
    cli_args = parse_args()

    train_dataset = load_train_dataset(cli_args.data_cache_dir)
    print(train_dataset)
    print(train_dataset[0])

    model_shortname = cli_args.model_name.split("/")[-1]
    run_name = f"{model_shortname}-AgentIR-instruct-{cli_args.learning_rate}-{cli_args.batch_size}-{cli_args.mini_batch_size}"
    output_dir = f"output/{model_shortname}/{run_name}"

    model = models.ColBERT(
        model_name_or_path=cli_args.model_name,
        document_length=4096,
        query_length=8192,
    )

    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=cli_args.mini_batch_size,
        temperature=cli_args.temperature,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cli_args.num_train_epochs,
        per_device_train_batch_size=cli_args.batch_size,
        per_device_eval_batch_size=cli_args.batch_size,
        save_steps=50,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=cli_args.learning_rate,
        dataloader_num_workers=8,
        accelerator_config={"split_batches": True},
    )

    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=model.tokenize, num_negatives=cli_args.num_negatives
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
