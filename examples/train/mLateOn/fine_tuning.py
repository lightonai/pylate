"""
Contrastive fine-tuning of a ColBERT model on the NV-Embed KD dataset
(query + 1 positive + stored hard negatives).

What to tweak
-------------
CLI:
  --model_name        Checkpoint to fine-tune (typically the pretraining
                      output).
  --learning_rate / --temperature / --num_train_epochs
  --batch_size        Also the pool of in-batch negatives.
  --document_length   Document token budget.

Data build (load_train_datasets — these are baked into the on-disk cache,
delete {cache_dir}/{split} after changing any of them):
  splits              Which subsets of the KD dataset to train on.
  nv_threshold=0.99   A document counts as a negative only if its KD score
                      is < 0.99 * positive score (false-negative filter;
                      lower = stricter).
  num_negatives=50    Hard negatives stored per query.

In main():
  num_negatives=7 (collator)  Negatives actually sampled per step from the
                              stored 50.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from typing import Callable

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from pylate import evaluation, losses, models

set_seed(42)


class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.stop_at_step:
            print(f"\n Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class KDToContrastive:
    """Converts a KD dataset into a contrastive one with query-positive-negatives format."""

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        num_negatives: int = 32,
        nv_threshold: float = 0.95,
    ) -> None:
        if isinstance(queries, datasets.DatasetDict):
            self.queries = queries[split]
        else:
            self.queries = queries

        if isinstance(documents, datasets.DatasetDict):
            self.documents = documents[split]
        else:
            self.documents = documents

        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold

        self.queries_index = {
            query_id: i for i, query_id in enumerate(iterable=self.queries["query_id"])
        }
        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(iterable=self.documents["document_id"])
        }

    def has_enough_negatives(self, example):
        scores = example["scores"]
        positive_score = scores[0]
        count = sum(
            1 for score in scores[1:] if score < self.nv_threshold * positive_score
        )
        return count >= self.num_negatives

    def map_to_query_positive_negatives(self, example):
        query_id = example["query_id"]
        document_ids = example["document_ids"]
        scores = example["scores"]

        query_text = self.queries[self.queries_index[query_id]]["query"]
        positive_id = document_ids[0]
        positive_text = self.documents[self.documents_index[positive_id]]["document"]
        positive_score = scores[0]

        row = {"query": query_text, "positive": positive_text}

        total_negatives = 0
        for i in range(1, len(document_ids)):
            if scores[i] < self.nv_threshold * positive_score:
                negative_id = document_ids[i]
                row[f"negative_{total_negatives}"] = self.documents[
                    self.documents_index[negative_id]
                ]["document"]
                total_negatives += 1
                if total_negatives >= self.num_negatives:
                    break

        return row


def load_train_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    splits = ["trivia", "hotpotqa", "nq", "msmarco", "fever", "squadv2", "fiqa"]

    for split in splits:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            print("Loaded dataset from disk")
        except FileNotFoundError:
            print("Creating dataset")
            dataset = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="scores",
                num_proc=144,
                split=split,
            )
            queries = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="queries",
                num_proc=144,
                split=split,
            )
            documents = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="documents",
                num_proc=144,
                split=split,
            )
            # Baked into the cache — delete {cache_dir}/{split} after changing
            processor = KDToContrastive(
                queries, documents, num_negatives=50, nv_threshold=0.99
            )
            dataset = dataset.filter(
                processor.has_enough_negatives,
                desc="Filtering examples with <50 negatives",
            ).map(
                processor.map_to_query_positive_negatives,
                remove_columns=dataset.column_names,
                desc="Creating query-positive-negatives dataset",
            )
            dataset.save_to_disk(f"{cache_dir}/{split}")
        train_dataset[split] = dataset
    return train_dataset


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT that randomly samples a subset of negative columns per batch."""

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives

        if valid_label_columns is None:
            valid_label_columns = ["label", "scores"]
        self.valid_label_columns = valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        for label_column in self.valid_label_columns:
            if label_column in columns:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                columns.remove(label_column)
                break

        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
            columns_to_process = other_columns + sampled_negatives
        else:
            columns_to_process = columns

        for column in columns_to_process:
            if "_id" not in column:
                is_query = "query" in column or "anchor" in column
                texts = [row[column] for row in features]
                if isinstance(texts[0], list):
                    texts = list(itertools.chain(*texts))
                tokenized = self.tokenize_fn(
                    texts,
                    is_query=is_query,
                    pad=True,
                )
                for key, value in tokenized.items():
                    batch[f"{column}_{key}"] = value

        return batch


def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(
        description="Fine-tune ColBERT with configurable hyperparameters"
    )

    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--document_length", type=int, default=300)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=-1,
        help="Stop training after this many steps (set to -1 to disable)",
    )
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache/nv_embed_cached_supervised_99",
    )

    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    # Build/cache the datasets on the main process only; other ranks wait,
    # then load the cached version from disk.
    with accelerator.main_process_first():
        train_dataset = load_train_datasets(cache_dir=args.cache_dir)
    print(train_dataset)

    model_shortname = args.model_name.split("/")[-1]

    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    temp_str = str(args.temperature).replace(".", "")
    run_name = (
        f"ColBERT-{model_shortname}-finetune-"
        f"lr{lr_str}-temp{temp_str}-"
        f"bs{args.batch_size}-"
        f"nv-embed-0.99-7negs"
    )

    output_dir = f"output/{model_shortname}/{run_name}"

    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Document Length: {args.document_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Stop at Step: {args.stop_at_step if args.stop_at_step > 0 else 'Disabled'}")
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 60}\n")

    model = models.ColBERT(args.model_name, document_length=args.document_length)

    dev_evaluator = evaluation.NanoBEIREvaluator()
    train_loss = losses.Contrastive(model, temperature=args.temperature)

    callbacks = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        fp16=False,
        bf16=True,
        seed=42,
        run_name=run_name,
        learning_rate=args.learning_rate,
        dataloader_num_workers=8,
        accelerator_config={
            "split_batches": True,
        },
    )

    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=model.tokenize,
        num_negatives=7,  # sampled per step from the stored 50
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=callbacks if callbacks else None,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model saved to: {output_dir}/final")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
