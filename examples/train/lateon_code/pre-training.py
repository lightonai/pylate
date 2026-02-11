from __future__ import annotations

import argparse
import itertools
import os
from typing import Callable

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers

from pylate import losses, models


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT model with optimized per-row negative sampling.

    Parameters
    ----------
    tokenize_fn
        The function to tokenize the input text.
    valid_label_columns
        The name of the columns that contain the labels: scores or labels.
    num_negatives
        Number of negatives to sample from available negative columns.
    temperature
        Temperature parameter for softmax sampling. Higher values (>1) make
        the distribution more uniform, lower values (<1) make it more peaked.
        Default is 1.0 (standard softmax).
    """

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
        temperature: float = 0.05,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        self.temperature = temperature

        if valid_label_columns is None:
            valid_label_columns = ["label", "scores"]

        self.valid_label_columns = valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of features into a batch."""
        batch = {"return_loss": True}

        columns = list(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the target label.
        for label_column in self.valid_label_columns:
            if label_column in columns:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                columns.remove(label_column)
                break

        # Separate negative columns from other columns
        negative_columns = [
            col
            for col in columns
            if col.startswith("negative_") and col != "negative_scores"
        ]

        # Collect query and document texts for all rows
        query_texts = [
            row["query"] if "query" in row else row.get("anchor", "")
            for row in features
        ]
        document_texts = [row["document"] for row in features]

        # For each row, sample negatives and collect texts
        all_negative_texts = []

        if (
            self.num_negatives is not None
            and negative_columns
            and "negative_scores" in features[0]
        ):
            k = min(self.num_negatives, len(negative_columns))

            # Vectorized softmax with temperature for all rows
            all_scores = np.array(
                [row["negative_scores"] for row in features], dtype=np.float32
            )
            # Apply temperature scaling
            scaled_scores = all_scores / self.temperature
            exp_scores = np.exp(
                scaled_scores - scaled_scores.max(axis=1, keepdims=True)
            )
            all_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Sample and collect negative texts for each row
            for i, row in enumerate(features):
                sampled_indices = np.random.choice(
                    len(negative_columns), size=k, replace=False, p=all_probs[i]
                )
                row_negatives = [row[negative_columns[idx]] for idx in sampled_indices]
                all_negative_texts.append(row_negatives)
        elif self.num_negatives is not None and negative_columns:
            # Uniform sampling per row
            k = min(self.num_negatives, len(negative_columns))
            for row in features:
                sampled_indices = np.random.choice(
                    len(negative_columns), size=k, replace=False
                )
                row_negatives = [row[negative_columns[idx]] for idx in sampled_indices]
                all_negative_texts.append(row_negatives)

        # Tokenize queries
        tokenized = self.tokenize_fn(query_texts, is_query=True, pad=True)
        for key, value in tokenized.items():
            batch[f"query_{key}"] = value

        # Tokenize documents
        tokenized = self.tokenize_fn(document_texts, is_query=False, pad=True)
        for key, value in tokenized.items():
            batch[f"document_{key}"] = value

        # Tokenize negatives - flatten all negative texts and tokenize in one go
        if all_negative_texts:
            flat_negatives = list(itertools.chain(*all_negative_texts))
            tokenized = self.tokenize_fn(flat_negatives, is_query=False, pad=True)

            # Reshape back to (batch_size, num_negatives, seq_len)
            batch_size = len(features)
            num_negs = len(all_negative_texts[0])

            for key, value in tokenized.items():
                # value shape: (batch_size * num_negatives, seq_len)
                # Reshape to: (batch_size, num_negatives, seq_len)
                reshaped = value.view(batch_size, num_negs, -1)
                # Flatten to: (batch_size * num_negatives, seq_len) for the model
                batch[f"negative_{key}"] = reshaped.view(batch_size * num_negs, -1)

        return batch


def load_train_datasets():
    """Load all available splits from CornStack, with caching"""
    cache_dir = "./cache_data"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    splits = [
        "python",
        "php",
        "go",
        "ruby",
        "javascript",
        "java",
    ]
    for split in splits:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            print(f"Loaded {split}")
        except:
            print(f"{split} missing")

            dataset = load_dataset(
                "lightonai/cornstack", name=split, split="train", num_proc=144
            )
            dataset.save_to_disk(f"{cache_dir}/{split}", num_proc=100)
            print(f"Loaded {split} dataset with {len(dataset)} examples.")

        train_dataset[split] = dataset
    return train_dataset


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_dataset = load_train_datasets()
    print(train_dataset)

    # Define training parameters
    num_train_epochs = 1
    lr = 6e-5
    batch_size = 128
    document_length = 2048
    query_length = 256
    mini_batch_size = 128
    model_name = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    model_shortname = model_name.split("/")[-1]

    # Set run name and output directory
    run_name = f"{model_shortname}-bs-{batch_size}-lr-{lr}-doclen-{document_length}-querylen-{query_length}-epoch-{num_train_epochs}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
        skiplist_words=[],
    )

    # Setup evaluation and loss
    # Needs to be merged first
    # dev_evaluator = evaluation.CodeSearchNetworkEvaluator()
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=mini_batch_size,
        temperature=0.07,
    )

    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,  # * dist.get_world_size(),  # We multiply by the world size because we are using split_batches
        per_device_eval_batch_size=batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        dataloader_num_workers=8,
        accelerator_config={
            "split_batches": True,
        },
    )
    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=model.tokenize, num_negatives=15
    )
    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        # evaluator=dev_evaluator,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
