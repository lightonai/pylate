from __future__ import annotations

import argparse
import itertools
import os
import random
from itertools import accumulate
from typing import Callable

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import MultiDatasetDefaultBatchSampler
from sentence_transformers.training_args import MultiDatasetBatchSamplers

from pylate import losses, models


class MultinomialBatchSampler(MultiDatasetDefaultBatchSampler):
    """Batch sampler that samples from each dataset according to multinomial sampling with temperature."""

    def __init__(
        self,
        dataset,
        batch_samplers,
        alpha=0.5,
        generator=None,
        seed=0,
        steps_per_epoch=32002,
    ):
        super().__init__(dataset, batch_samplers, generator, seed)
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        weights = [n**self.alpha for n in num_samples]
        total_weight = sum(weights)
        self.sampling_probs = [w / total_weight for w in weights]

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        sample_offsets = [0] + list(
            accumulate([len(dataset) for dataset in self.dataset.datasets])
        )
        reset_counts = [0] * len(self.batch_samplers)
        if self.steps_per_epoch is not None:
            total_batches = self.steps_per_epoch
        else:
            total_batches = sum(len(sampler) for sampler in self.batch_samplers)
        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        probs_tensor = torch.tensor(self.sampling_probs, dtype=torch.float32)
        for _ in range(total_batches):
            dataset_idx = torch.multinomial(
                probs_tensor, num_samples=1, replacement=True, generator=self.generator
            ).item()
            sample_offset = sample_offsets[dataset_idx]
            try:
                batch = next(batch_samplers[dataset_idx])
            except StopIteration:
                reset_counts[dataset_idx] += 1
                sampler = self.batch_samplers[dataset_idx]
                if hasattr(sampler, "sampler") and hasattr(
                    sampler.sampler, "generator"
                ):
                    new_seed = (
                        self.seed + self.epoch * 100000 + reset_counts[dataset_idx]
                    )
                    sampler.sampler.generator.manual_seed(new_seed)
                elif hasattr(sampler, "sampler") and hasattr(
                    sampler.sampler, "set_epoch"
                ):
                    sampler.sampler.set_epoch(
                        self.epoch * 100000 + reset_counts[dataset_idx]
                    )
                batch_samplers[dataset_idx] = iter(sampler)
                batch = next(batch_samplers[dataset_idx])
            yield [idx + sample_offset for idx in batch]

    def __len__(self) -> int:
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch
        return sum(len(sampler) for sampler in self.batch_samplers)


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT model with negative sampling and prompt support."""

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
        prompts: dict[str, str] | dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        self.prompts = prompts if prompts is not None else {}
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
        prompts = self._resolve_prompts(batch)
        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]
        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
            columns_to_process = other_columns + sampled_negatives
        else:
            columns_to_process = columns
        for column in columns_to_process:
            if "_id" in column:
                continue
            is_query = "query" in column or "anchor" in column
            texts = [row[column] for row in features]
            if isinstance(texts[0], list):
                texts = list(itertools.chain(*texts))
            prompt = self._get_prompt_for_column(column, prompts)
            if prompt:
                try:
                    inputs = [prompt + row[column] for row in features]
                except:
                    key = "document" if "document" in features[0][column] else "query"
                    inputs = [prompt + row[column][key] for row in features]
            else:
                inputs = [row[column] for row in features]
            tokenized = self.tokenize_fn(inputs, is_query=is_query, pad=True)
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value
        return batch

    def _resolve_prompts(self, batch: dict) -> dict[str, str]:
        prompts = self.prompts
        if not prompts or not isinstance(prompts, dict):
            return {}
        first_value = next(iter(prompts.values()), None)
        is_nested = isinstance(first_value, dict)
        if is_nested:
            is_multi_dataset = "dataset_name" in batch
            if is_multi_dataset and batch["dataset_name"] in prompts:
                return prompts[batch["dataset_name"]]
            elif not is_multi_dataset:
                raise ValueError(
                    f"Nested prompts but no dataset_name in batch. Keys: {list(prompts.keys())!r}"
                )
            else:
                return {}
        return prompts

    def _get_prompt_for_column(
        self, column: str, prompts: dict[str, str]
    ) -> str | None:
        if not prompts:
            return None
        if column in prompts:
            return prompts[column]
        if column.startswith("negative_"):
            if "negative" in prompts:
                return prompts["negative"]
            if "documents" in prompts:
                return prompts["documents"]
        if column == "positive" and "documents" in prompts:
            return prompts["documents"]
        return None


class KDToContrastive:
    """Dataset processing class for converting a KD dataset into a contrastive one."""

    def __init__(
        self,
        queries,
        documents,
        split: str = "train",
        num_negatives: int = 32,
        nv_threshold: float = 0.95,
    ):
        self.queries = (
            queries[split] if isinstance(queries, datasets.DatasetDict) else queries
        )
        self.documents = (
            documents[split]
            if isinstance(documents, datasets.DatasetDict)
            else documents
        )
        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold
        self.queries_index = {qid: i for i, qid in enumerate(self.queries["query_id"])}
        self.documents_index = {
            did: i for i, did in enumerate(self.documents["document_id"])
        }

    def has_enough_negatives(self, example):
        scores = example["scores"]
        positive_score = scores[0]
        count = sum(
            1
            for score in scores[1:]
            if score < self.nv_threshold * positive_score and score != -1
        )
        return count >= self.num_negatives

    def map_to_query_positive_negatives(self, example):
        query_id, document_ids, scores = (
            example["query_id"],
            example["document_ids"],
            example["scores"],
        )
        query_text = self.queries[self.queries_index[query_id]]
        positive_id, positive_score = document_ids[0], scores[0]
        positive_text = self.documents[self.documents_index[positive_id]]
        row = {"query": query_text, "positive": positive_text}
        total_negatives = 0
        for i in range(1, len(document_ids)):
            if scores[i] < self.nv_threshold * positive_score and scores[i] != -1:
                row[f"negative_{total_negatives}"] = self.documents[
                    self.documents_index[document_ids[i]]
                ]
                total_negatives += 1
                if total_negatives >= self.num_negatives:
                    break
        return row


def load_train_datasets():
    cache_dir = "/home/antoine_chaffin/supervised_code_data_99"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    splits = [
        "apps",
        "synthetictext2sql",
        "cosqa",
        "codefeedbackst",
        "codefeedbackmt",
        "stackoverflowqa",
        "codetranscontest",
        "codetransdl",
        "CodeSearchNet_go",
        "CodeSearchNet_java",
        "CodeSearchNet_javascript",
        "CodeSearchNet_php",
        "CodeSearchNet_python",
        "CodeSearchNet_ruby",
        "CodeSearchNet_ccr_go",
        "CodeSearchNet_ccr_java",
        "CodeSearchNet_ccr_javascript",
        "CodeSearchNet_ccr_php",
        "CodeSearchNet_ccr_python",
        "CodeSearchNet_ccr_ruby",
    ]
    for split in splits:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            print(f"Loaded dataset from disk {split}")
        except FileNotFoundError:
            print(f"Creating dataset {split}")
            dataset = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup-code",
                name="scores",
                num_proc=45,
                split=split,
            )
            queries = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup-code",
                name="queries",
                num_proc=45,
                split=split,
            )
            documents = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup-code",
                name="documents",
                num_proc=45,
                split=split,
            )
            processor = KDToContrastive(
                queries, documents, num_negatives=50, nv_threshold=0.99
            )
            dataset = dataset.filter(
                processor.has_enough_negatives, desc="Filtering", num_proc=45
            ).map(
                processor.map_to_query_positive_negatives,
                remove_columns=dataset.column_names,
                desc="Creating",
                num_proc=45,
            )
            dataset.save_to_disk(f"{cache_dir}/{split}")
        train_dataset[split] = dataset
    return train_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-multinomial-sampler",
        action="store_true",
        default=False,
        help="Use MultinomialBatchSampler instead of PROPORTIONAL sampler",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=8e-6,
        help="Learning rate for training (default: 8e-6)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss (default: 0.07)",
    )
    args = parser.parse_args()

    train_dataset = load_train_datasets()
    print(train_dataset)

    num_train_epochs = 1
    lr = args.learning_rate
    batch_size = 128
    document_length = 2048
    query_length = 256

    model_name = "lightonai/GTE-ModernColBERT-v1"
    # Or your pretrained model

    model_shortname = model_name.split("/")[-1]

    sampler_name = "multinomial" if args.use_multinomial_sampler else "proportional"
    run_name = f"{model_shortname}-bs-{batch_size}-lr-{lr}-doclen-{document_length}-querylen-{query_length}-epoch-{num_train_epochs}-{sampler_name}-finetuning-temp-{args.temperature}"
    output_dir = f"output/{model_shortname}/{run_name}"

    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
        skiplist_words=[],
    )

    # Needs to be merged first
    # dev_evaluator = evaluation.CodeSearchNetworkEvaluator()
    train_loss = losses.Contrastive(
        model=model, gather_across_devices=True, temperature=args.temperature
    )

    # Select batch sampler based on argument
    if args.use_multinomial_sampler:
        multi_dataset_sampler = MultinomialBatchSampler
    else:
        multi_dataset_sampler = MultiDatasetBatchSamplers.PROPORTIONAL

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        multi_dataset_batch_sampler=multi_dataset_sampler,
        eval_strategy="steps",
        eval_steps=2500,
        save_steps=5000,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        dataloader_num_workers=8,
        accelerator_config={"split_batches": True},
    )

    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=model.tokenize, num_negatives=15
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
