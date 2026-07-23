"""
ColBERT pretraining script.

What to tweak
-------------
CLI:
  --model_name        Backbone checkpoint. If its hidden size is not 768,
                      update the Dense projection stack in main().
  --learning_rate     Peak LR (linear decay over --max_steps, 5% warmup).
  --temperature       Contrastive temperature.
  --batch_size        Global contrastive batch = pool of in-batch negatives,
                      so it changes task difficulty, not just throughput.
  --mini_batch_size   Cached-loss chunk size: pure GPU-memory knob, does NOT
                      affect results. Raise it until just below OOM.
  --query_length / --document_length
                      Per-side token budgets (+5 headroom is added for the
                      marker/prefix tokens).
  --max_steps         LR-schedule horizon; the streaming dataset is sized
                      to cover it.
  --stop_at_step      Where training actually stops; keeping it below
                      --max_steps stops before the LR fully decays.
  --dataset_id        HF dataset repo. Flat layout ({group}-{shard}-of-
                      {total}.parquet at the root) is grouped by prefix;
                      a repo with subdirectories (e.g. one folder per
                      language, like lightonai/contrastive-multilingual)
                      is grouped by folder. Each group is one sampled
                      dataset, and every batch comes from a single group.
  --multilingual_dataset_pattern / --languages
                      Optional per-language repos merged into the mix.
  --num_workers       DataLoader workers feeding the GPUs; raise it if GPUs
                      sit idle between steps. RAM grows with it (each worker
                      holds its own cursors and tables).
  --rg_window         Parquet row groups loaded and shuffled together per
                      read. Keep rg_window * rows_per_row_group comfortably
                      above batch_size so one batch doesn't need several
                      reads; larger also mixes each dataset better, at the
                      cost of RAM.
  --num_io_threads    Parallel row-group reads; only helps when a window
                      spans several files. Raise on slow/networked storage.
                      (None of these three change what the model learns —
                      only speed, RAM, and stream shuffling.)

In main():
  dense_1/2/3           Projection head: residual 768->1536->768 MLP, then a
                        768->128 output layer. dense_3 out_features sets the
                        per-token embedding dim; in_features must match the
                        backbone hidden size.
  alpha (DatasetConfig) Dataset sampling temperature: 0 = uniform across
                        datasets, 1 = proportional to dataset size.
  max_active_cursors    LRU cap on simultaneously open datasets (RAM vs
                        re-open cost).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Callable, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from datasets import Features, Value, load_dataset
from datasets import IterableDataset as HFIterableDataset
from huggingface_hub import HfApi, snapshot_download
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import NanoBEIREvaluator as NanoBEIREvaluatorST
from sentence_transformers.models import Transformer
from tqdm import tqdm
from transformers import TrainerCallback

from pylate import evaluation, losses, models, utils

logger = logging.getLogger(__name__)

set_seed(42)


def _log(msg: str):
    print(msg, flush=True)


def _stable_hash(s: str) -> int:
    """Deterministic hash consistent across processes and Python runs."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)


DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]

dataset_name_to_human_readable = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}

SUPPORTED_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "no", "pt", "sv"]


# ============================================================
# Data cursor: reads parquet row groups, pre-converts to lists
# ============================================================


class _DatasetCursor:
    """
    Reads row groups from parquet files for a single dataset.

    RG ordering is a pure function of (base_seed, dataset_name, global_position).
    Epoch = _global_rg_pos // n_rgs. Workers partition each epoch's shuffled list
    via shard_id::num_shards.
    """

    def __init__(
        self,
        file_metadata: list[tuple[str, int, int]],
        dataset_name: str,
        shard_id: int = 0,
        num_shards: int = 1,
        base_seed: int = 42,
        rg_window: int = 3,
        num_io_threads: int = 4,
        resume_pos: int = 0,
    ):
        self.file_metadata = file_metadata
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rg_window = rg_window
        self.num_io_threads = num_io_threads

        self._all_rgs = []
        for path, _, n_rgs in file_metadata:
            for rg_idx in range(n_rgs):
                self._all_rgs.append((path, rg_idx))

        self._n_rgs = len(self._all_rgs)
        self._rg_seed = base_seed + _stable_hash(dataset_name)

        self._partitioned = self._n_rgs >= num_shards
        if not self._partitioned and num_shards > 1:
            _log(
                f"[CURSOR shard={shard_id}] Dataset '{dataset_name}' has "
                f"{self._n_rgs} RGs < {num_shards} workers — sharing all RGs "
                f"(no partition)"
            )

        self._global_rg_pos = resume_pos

        self.table: pa.Table | None = None
        self._num_rows: int = 0
        self.pos = 0

    def _get_epoch_order(self, epoch: int) -> list[int]:
        order = list(range(self._n_rgs))
        rng = random.Random(self._rg_seed + epoch)
        rng.shuffle(order)
        return order

    def _next_rg_indices(self, count: int) -> list[int]:
        result = []
        pos = self._global_rg_pos

        while len(result) < count:
            epoch = pos // self._n_rgs
            offset_in_epoch = pos % self._n_rgs

            order = self._get_epoch_order(epoch)

            if self._partitioned:
                worker_order = order[self.shard_id :: self.num_shards]
                items_before = 0
                for i in range(offset_in_epoch):
                    if i % self.num_shards == self.shard_id:
                        items_before += 1
                worker_remaining = worker_order[items_before:]
            else:
                worker_remaining = order[offset_in_epoch:]

            take = min(count - len(result), len(worker_remaining))
            result.extend(worker_remaining[:take])

            if take == len(worker_remaining):
                pos = (epoch + 1) * self._n_rgs
            else:
                if self._partitioned:
                    consumed_global = 0
                    worker_count = 0
                    for i in range(offset_in_epoch, self._n_rgs):
                        consumed_global += 1
                        if i % self.num_shards == self.shard_id:
                            worker_count += 1
                            if worker_count == take:
                                break
                    pos += consumed_global
                else:
                    pos += take

        self._global_rg_pos = pos
        return result

    @staticmethod
    def _read_row_groups(path: str, rg_indices: list[int]) -> pa.Table:
        pf = pq.ParquetFile(path)
        return pf.read_row_groups(rg_indices, columns=["query", "document"])

    def _load_next_window(self, rng: random.Random):
        indices = self._next_rg_indices(self.rg_window)
        selected = [self._all_rgs[i] for i in indices]

        by_file: dict[str, list[int]] = defaultdict(list)
        for path, rg_idx in selected:
            by_file[path].append(rg_idx)

        if self.num_io_threads > 1 and len(by_file) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.num_io_threads, len(by_file))
            ) as pool:
                tables = list(
                    pool.map(
                        lambda item: self._read_row_groups(item[0], item[1]),
                        by_file.items(),
                    )
                )
        else:
            tables = [self._read_row_groups(p, rgs) for p, rgs in by_file.items()]

        table = pa.concat_tables(tables)
        del tables
        new_fields = []
        for field in table.schema:
            if field.type == pa.string():
                new_fields.append(field.with_type(pa.large_string()))
            else:
                new_fields.append(field)
        table = table.cast(pa.schema(new_fields))

        n = table.num_rows
        perm = np.random.RandomState(rng.randint(0, 2**31)).permutation(n)
        self.table = table.take(perm)
        self._num_rows = n
        self.pos = 0

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> tuple[list[str], list[str]]:
        queries: list[str] = []
        documents: list[str] = []
        remaining = batch_size

        while remaining > 0:
            if self.table is None or self.pos >= self._num_rows:
                self._load_next_window(rng)

            n = min(remaining, self._num_rows - self.pos)
            sliced = self.table.slice(self.pos, n)
            queries.extend(sliced.column("query").to_pylist())
            documents.extend(sliced.column("document").to_pylist())

            self.pos += n
            remaining -= n

        return queries, documents

    def release(self):
        self.table = None
        self._num_rows = 0
        self.pos = 0


# ============================================================
# Module-level generator for HF multi-shard support
# ============================================================


def _iterate_sharded(
    shard_id,
    num_shards,
    file_metadata,
    dataset_names,
    sampling_probs,
    batch_size,
    seed,
    steps_per_epoch,
    rg_window,
    max_active_cursors,
    num_io_threads,
):
    dataset_seed = seed
    row_seed = seed + shard_id * 7919

    _log(
        f"[SHARD {shard_id}/{num_shards}] dataset_seed={dataset_seed}, "
        f"row_seed={row_seed}"
    )

    dataset_names = list(dataset_names)
    sampling_probs = list(sampling_probs)

    dataset_generator = torch.Generator()
    dataset_generator.manual_seed(dataset_seed)
    probs_tensor = torch.tensor(sampling_probs, dtype=torch.float32)

    rng = random.Random(row_seed)
    cursors: OrderedDict[str, _DatasetCursor] = OrderedDict()

    cursor_positions: dict[str, int] = {}

    for step_idx in range(steps_per_epoch):
        idx = torch.multinomial(
            probs_tensor, 1, replacement=True, generator=dataset_generator
        ).item()
        name = dataset_names[idx]

        if name in cursors:
            cursors.move_to_end(name)
        else:
            if max_active_cursors > 0 and len(cursors) >= max_active_cursors:
                evicted_name, evicted = cursors.popitem(last=False)
                cursor_positions[evicted_name] = evicted._global_rg_pos
                evicted.release()

            cursors[name] = _DatasetCursor(
                file_metadata[name],
                dataset_name=name,
                shard_id=shard_id,
                num_shards=num_shards,
                base_seed=seed,
                rg_window=rg_window,
                num_io_threads=num_io_threads,
                resume_pos=cursor_positions.get(name, 0),
            )

        queries, documents = cursors[name].sample(batch_size, rng)
        for q, d in zip(queries, documents):
            yield {"query": q, "positive": d}

    for c in cursors.values():
        c.release()


# ============================================================
# Dataset config holder (reads metadata, computes sampling probs)
# ============================================================


class DatasetConfig:
    def __init__(
        self,
        dataset_file_map: dict[str, list[str]],
        batch_size: int = 16384,
        alpha: float = 0.5,
        seed: int = 42,
        steps_per_epoch: int = 110_000,
        rg_window: int = 3,
        max_active_cursors: int = 0,
        num_io_threads: int = 4,
        num_workers: int = 1,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.rg_window = rg_window
        self.max_active_cursors = max_active_cursors
        self.num_io_threads = num_io_threads
        self.num_workers = num_workers
        self.dataset_names = list(dataset_file_map.keys())

        self.file_metadata: dict[str, list[tuple[str, int, int]]] = {}
        dataset_sizes: dict[str, int] = {}
        for name, files in dataset_file_map.items():
            meta = []
            total = 0
            for f in files:
                m = pq.read_metadata(f)
                meta.append((f, m.num_rows, m.num_row_groups))
                total += m.num_rows
            self.file_metadata[name] = meta
            dataset_sizes[name] = total

        weights = [dataset_sizes[n] ** alpha for n in self.dataset_names]
        total_w = sum(weights)
        self.sampling_probs = [w / total_w for w in weights]

        _log(
            f"[DATASET] {len(dataset_file_map)} datasets, "
            f"total {sum(dataset_sizes.values()):,} rows"
        )
        _log(
            f"[DATASET] rg_window={rg_window}, io_threads={num_io_threads}, "
            f"num_workers={num_workers}"
        )

    def gen_kwargs_for_shard(self, shard_id: int) -> dict:
        return {
            "shard_id": shard_id,
            "num_shards": self.num_workers,
            "file_metadata": self.file_metadata,
            "dataset_names": tuple(self.dataset_names),
            "sampling_probs": tuple(self.sampling_probs),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "steps_per_epoch": self.steps_per_epoch,
            "rg_window": self.rg_window,
            "max_active_cursors": self.max_active_cursors,
            "num_io_threads": self.num_io_threads,
        }


# ============================================================
# PyTorch IterableDataset wrapper (bypasses HF n_shards check)
# ============================================================


class MultiWorkerDataset(HFIterableDataset):
    def __init__(self, hf_dataset: HFIterableDataset, ds_config: DatasetConfig):
        self.__dict__.update(hf_dataset.__dict__)
        self._ds_config = ds_config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard_id = 0 if worker_info is None else worker_info.id
        _log(f"[WORKER {shard_id}] Starting iterator")
        yield from _iterate_sharded(**self._ds_config.gen_kwargs_for_shard(shard_id))


# ============================================================
# Callbacks
# ============================================================


class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_at_step:
            _log(f"\n>>> Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class MultilingualNanoBEIREvaluator(NanoBEIREvaluatorST):
    """Evaluate ColBERT models on the Multilingual NanoBEIR collection."""

    def __init__(
        self,
        dataset_names: list[DatasetNameType] | None = None,
        languages: list[str] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 512,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        dataset_path: str | None = None,
    ):
        if dataset_names is None:
            dataset_names = list(dataset_name_to_human_readable.keys())
        if languages is None:
            languages = SUPPORTED_LANGUAGES

        self.dataset_names = dataset_names
        self.languages = languages
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.write_csv = write_csv
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.dataset_path = dataset_path or "lightonai/nanobeir-multilingual"

        self.name = f"MultilingualNanoBEIR_{aggregate_key}"

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self._validate_dataset_names()
        self._validate_languages()
        self._validate_prompts()

        ir_evaluator_kwargs = {
            "mrr_at_k": mrr_at_k,
            "ndcg_at_k": ndcg_at_k,
            "accuracy_at_k": accuracy_at_k,
            "precision_recall_at_k": precision_recall_at_k,
            "map_at_k": map_at_k,
            "show_progress_bar": show_progress_bar,
            "batch_size": batch_size,
            "write_csv": write_csv,
        }

        self.evaluators = []
        self.evaluator_index = {}
        for lang in self.languages:
            for name in self.dataset_names:
                try:
                    evaluator = self._load_dataset(name, lang, **ir_evaluator_kwargs)
                    self.evaluators.append(evaluator)
                    self.evaluator_index[(name, lang)] = evaluator
                except Exception as e:
                    logger.warning(f"Failed to load {name}-{lang}: {e}")

        if not self.evaluators:
            raise ValueError(
                "No evaluators created. Please check your dataset_names and languages."
            )

        self.csv_file = f"MultilingualNanoBEIR_PyLate_{aggregate_key}_results.csv"
        self.csv_headers = self._build_csv_headers()

    def _build_csv_headers(self) -> list[str]:
        headers = ["epoch", "steps"]
        for k in self.accuracy_at_k:
            headers.append(f"MaxSim_accuracy@{k}")
        for k in self.precision_recall_at_k:
            headers.append(f"MaxSim_precision@{k}")
            headers.append(f"MaxSim_recall@{k}")
        for k in self.mrr_at_k:
            headers.append(f"MaxSim_mrr@{k}")
        for k in self.ndcg_at_k:
            headers.append(f"MaxSim_ndcg@{k}")
        for k in self.map_at_k:
            headers.append(f"MaxSim_map@{k}")
        return headers

    def __call__(
        self,
        model,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        per_language_results = {lang: {} for lang in self.languages}

        out_txt = ""
        if epoch != -1:
            out_txt = (
                f" after epoch {epoch}"
                if steps == -1
                else f" in epoch {epoch} after {steps} steps"
            )

        logger.info(
            f"Multilingual NanoBEIR Evaluation on {len(self.evaluators)} dataset-language pairs{out_txt}"
        )

        for evaluator in tqdm(
            self.evaluators,
            desc="Evaluating datasets",
            disable=not self.show_progress_bar,
        ):
            logger.info(f"Evaluating {evaluator.name}")
            results = evaluator(model, output_path, epoch, steps)

            lang = evaluator.name.split("-")[-1]

            for full_key, metric_value in results.items():
                parts = full_key.split("_")
                metric = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[full_key] = metric_value
                per_metric_results[metric].append(metric_value)

                if lang in per_language_results:
                    if metric not in per_language_results[lang]:
                        per_language_results[lang][metric] = []
                    per_language_results[lang][metric].append(metric_value)

        agg_results = {}
        for metric, values in per_metric_results.items():
            agg_results[f"{self.name}_{metric}"] = self.aggregate_fn(values)

        per_lang_agg_results = {}
        for lang in self.languages:
            for metric, values in per_language_results[lang].items():
                if values:
                    per_lang_agg_results[f"{self.name}_{lang}_{metric}"] = (
                        self.aggregate_fn(values)
                    )

        if output_path is not None and self.write_csv:
            self._write_csv(output_path, epoch, steps, agg_results)
            self._write_per_language_csv(
                output_path, epoch, steps, per_lang_agg_results
            )

        self._log_results(agg_results)
        self._log_per_language_results(per_lang_agg_results)

        if not hasattr(self, "primary_metric") or not self.primary_metric:
            self.primary_metric = f"{self.name}_MaxSim_ndcg@{max(self.ndcg_at_k)}"

        per_dataset_results.update(agg_results)
        per_dataset_results.update(per_lang_agg_results)
        return per_dataset_results

    def _write_csv(self, output_path, epoch, steps, agg_results):
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, self.csv_file)
        if not os.path.isfile(csv_path):
            with open(csv_path, mode="w", encoding="utf-8") as f:
                f.write(",".join(self.csv_headers) + "\n")
        with open(csv_path, mode="a", encoding="utf-8") as f:
            output_data = [epoch, steps]
            for k in self.accuracy_at_k:
                output_data.append(
                    agg_results.get(f"{self.name}_MaxSim_accuracy@{k}", "")
                )
            for k in self.precision_recall_at_k:
                output_data.append(
                    agg_results.get(f"{self.name}_MaxSim_precision@{k}", "")
                )
                output_data.append(
                    agg_results.get(f"{self.name}_MaxSim_recall@{k}", "")
                )
            for k in self.mrr_at_k:
                output_data.append(agg_results.get(f"{self.name}_MaxSim_mrr@{k}", ""))
            for k in self.ndcg_at_k:
                output_data.append(agg_results.get(f"{self.name}_MaxSim_ndcg@{k}", ""))
            for k in self.map_at_k:
                output_data.append(agg_results.get(f"{self.name}_MaxSim_map@{k}", ""))
            f.write(",".join(map(str, output_data)) + "\n")

    def _write_per_language_csv(self, output_path, epoch, steps, per_lang_results):
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(
            output_path,
            f"MultilingualNanoBEIR_PyLate_{self.aggregate_key}_per_language.csv",
        )
        headers = ["epoch", "steps", "language"]
        for k in self.accuracy_at_k:
            headers.append(f"MaxSim_accuracy@{k}")
        for k in self.precision_recall_at_k:
            headers.append(f"MaxSim_precision@{k}")
            headers.append(f"MaxSim_recall@{k}")
        for k in self.mrr_at_k:
            headers.append(f"MaxSim_mrr@{k}")
        for k in self.ndcg_at_k:
            headers.append(f"MaxSim_ndcg@{k}")
        for k in self.map_at_k:
            headers.append(f"MaxSim_map@{k}")
        if not os.path.isfile(csv_path):
            with open(csv_path, mode="w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
        with open(csv_path, mode="a", encoding="utf-8") as f:
            for lang in self.languages:
                output_data = [epoch, steps, lang]
                for k in self.accuracy_at_k:
                    output_data.append(
                        per_lang_results.get(
                            f"{self.name}_{lang}_MaxSim_accuracy@{k}", ""
                        )
                    )
                for k in self.precision_recall_at_k:
                    output_data.append(
                        per_lang_results.get(
                            f"{self.name}_{lang}_MaxSim_precision@{k}", ""
                        )
                    )
                    output_data.append(
                        per_lang_results.get(
                            f"{self.name}_{lang}_MaxSim_recall@{k}", ""
                        )
                    )
                for k in self.mrr_at_k:
                    output_data.append(
                        per_lang_results.get(f"{self.name}_{lang}_MaxSim_mrr@{k}", "")
                    )
                for k in self.ndcg_at_k:
                    output_data.append(
                        per_lang_results.get(f"{self.name}_{lang}_MaxSim_ndcg@{k}", "")
                    )
                for k in self.map_at_k:
                    output_data.append(
                        per_lang_results.get(f"{self.name}_{lang}_MaxSim_map@{k}", "")
                    )
                f.write(",".join(map(str, output_data)) + "\n")

    def _log_results(self, agg_results):
        logger.info(f"\nAggregated Results ({self.aggregate_key}):")
        for k in self.accuracy_at_k:
            val = agg_results.get(f"{self.name}_MaxSim_accuracy@{k}", 0)
            logger.info(f"  Accuracy@{k}: {val * 100:.2f}%")
        for k in self.precision_recall_at_k:
            prec = agg_results.get(f"{self.name}_MaxSim_precision@{k}", 0)
            rec = agg_results.get(f"{self.name}_MaxSim_recall@{k}", 0)
            logger.info(f"  Precision@{k}: {prec * 100:.2f}%")
            logger.info(f"  Recall@{k}: {rec * 100:.2f}%")
        for k in self.mrr_at_k:
            val = agg_results.get(f"{self.name}_MaxSim_mrr@{k}", 0)
            logger.info(f"  MRR@{k}: {val:.4f}")
        for k in self.ndcg_at_k:
            val = agg_results.get(f"{self.name}_MaxSim_ndcg@{k}", 0)
            logger.info(f"  NDCG@{k}: {val:.4f}")
        for k in self.map_at_k:
            val = agg_results.get(f"{self.name}_MaxSim_map@{k}", 0)
            logger.info(f"  MAP@{k}: {val:.4f}")

    def _log_per_language_results(self, per_lang_results):
        logger.info(f"\nPer-Language Results ({self.aggregate_key}):")
        for lang in self.languages:
            logger.info(f"\n  Language: {lang}")
            for k in self.accuracy_at_k:
                val = per_lang_results.get(f"{self.name}_{lang}_MaxSim_accuracy@{k}", 0)
                logger.info(f"    Accuracy@{k}: {val * 100:.2f}%")
            for k in self.precision_recall_at_k:
                prec = per_lang_results.get(
                    f"{self.name}_{lang}_MaxSim_precision@{k}", 0
                )
                rec = per_lang_results.get(f"{self.name}_{lang}_MaxSim_recall@{k}", 0)
                logger.info(f"    Precision@{k}: {prec * 100:.2f}%")
                logger.info(f"    Recall@{k}: {rec * 100:.2f}%")
            for k in self.mrr_at_k:
                val = per_lang_results.get(f"{self.name}_{lang}_MaxSim_mrr@{k}", 0)
                logger.info(f"    MRR@{k}: {val:.4f}")
            for k in self.ndcg_at_k:
                val = per_lang_results.get(f"{self.name}_{lang}_MaxSim_ndcg@{k}", 0)
                logger.info(f"    NDCG@{k}: {val:.4f}")
            for k in self.map_at_k:
                val = per_lang_results.get(f"{self.name}_{lang}_MaxSim_map@{k}", 0)
                logger.info(f"    MAP@{k}: {val:.4f}")

    def _get_human_readable_name(self, dataset_name, language):
        return f"Nano{dataset_name_to_human_readable[dataset_name.lower()]}-{language}"

    def _load_dataset(self, dataset_name, language, **ir_evaluator_kwargs):
        base_name = f"Nano{dataset_name_to_human_readable[dataset_name.lower()]}"
        try:
            corpus = load_dataset(
                self.dataset_path, f"{base_name}_{language}", split="corpus"
            )
            queries = load_dataset(
                self.dataset_path, f"{base_name}_{language}", split="queries"
            )
            qrels = load_dataset(self.dataset_path, base_name, split="qrels")
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset {base_name} for language {language}: {e}"
            )

        corpus_dict = {
            sample["_id"]: sample["text"] for sample in corpus if sample.get("text")
        }
        queries_dict = {
            sample["_id"]: sample["text"] for sample in queries if sample.get("text")
        }
        qrels_dict = {}
        for sample in qrels:
            q_id = sample["query-id"]
            if q_id not in qrels_dict:
                qrels_dict[q_id] = set()
            qrels_dict[q_id].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(
                dataset_name, None
            )
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(
                dataset_name, None
            )

        human_readable_name = self._get_human_readable_name(dataset_name, language)
        return evaluation.PyLateInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        if not self.dataset_names:
            raise ValueError(
                "dataset_names cannot be empty. Use None for all datasets."
            )
        missing = [
            n
            for n in self.dataset_names
            if n.lower() not in dataset_name_to_human_readable
        ]
        if missing:
            raise ValueError(
                f"Dataset(s) {missing} not found. Valid: {list(dataset_name_to_human_readable.keys())}"
            )

    def _validate_languages(self):
        if not self.languages:
            raise ValueError(
                "languages cannot be empty. Use None for all supported languages."
            )
        unsupported = [
            language
            for language in self.languages
            if language not in SUPPORTED_LANGUAGES
        ]
        if unsupported:
            raise ValueError(
                f"Language(s) {unsupported} not supported. Supported: {SUPPORTED_LANGUAGES}"
            )

    def _validate_prompts(self):
        if isinstance(self.query_prompts, str):
            self.query_prompts = {
                name: self.query_prompts for name in self.dataset_names
            }
        elif self.query_prompts:
            missing = [n for n in self.dataset_names if n not in self.query_prompts]
            if missing:
                raise ValueError(f"Missing query prompts for: {missing}")
        if isinstance(self.corpus_prompts, str):
            self.corpus_prompts = {
                name: self.corpus_prompts for name in self.dataset_names
            }
        elif self.corpus_prompts:
            missing = [n for n in self.dataset_names if n not in self.corpus_prompts]
            if missing:
                raise ValueError(f"Missing corpus prompts for: {missing}")


# ============================================================
# Helpers
# ============================================================


def _download_hf_parquets(dataset_id, cache_dir):
    """Download parquet files from a HF Hub dataset repo and group them into datasets.

    Two layouts are supported (auto-detected):
    - Repo with subdirectories (e.g. one folder per language, like
      lightonai/contrastive-multilingual): grouped by top-level folder name.
    - Flat repo with {group}-{shard}-of-{total}.parquet at the root (like
      lightonai/embeddings-pre-training-curated): grouped by the prefix
      before the first '-'.
    """
    api = HfApi()
    repo_files = api.list_repo_files(dataset_id, repo_type="dataset")
    parquet_files = [f for f in repo_files if f.endswith(".parquet")]
    use_dirs = any("/" in pf for pf in parquet_files)

    # Parallel download of the whole repo (same engine and cache as the CLI,
    # so a manual pre-fetch is picked up here without re-downloading):
    #   hf download {dataset_id} --repo-type dataset --include "*.parquet" \
    #       --cache-dir {cache_dir}
    # For maximum throughput: pip install hf_transfer and
    # export HF_HUB_ENABLE_HF_TRANSFER=1 before launching.
    _log(
        f"[DATASET] Fetching {len(parquet_files)} parquet files from "
        f"{dataset_id} (cached files are reused)"
    )
    snapshot_path = snapshot_download(
        dataset_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=["*.parquet"],
        max_workers=16,
    )

    file_map: dict[str, list[str]] = {}
    for pf in parquet_files:
        if use_dirs:
            group = pf.split("/")[0] if "/" in pf else "default"
        else:
            group = pf.split("-")[0]
        file_map.setdefault(group, []).append(os.path.join(snapshot_path, pf))

    return file_map


def build_dataset_file_map(
    dataset_id, cache_dir, multilingual_pattern=None, languages=None
):
    """Download parquets from HF Hub, optionally merging per-language repos.

    Each language has its own HF dataset (one repo per language), and configs
    within each repo are subsplits (e.g. agnews, amazon_reviews, ...).
    """
    dataset_file_map = _download_hf_parquets(dataset_id, cache_dir)

    if multilingual_pattern and languages:
        for lang in languages:
            lang_dataset_id = multilingual_pattern.format(lang=lang)
            _log(f"[DATASET] Downloading multilingual: {lang_dataset_id}")
            lang_map = _download_hf_parquets(lang_dataset_id, cache_dir)
            for config, files in lang_map.items():
                dataset_file_map[f"{config}_{lang}"] = files

    return dataset_file_map


# ============================================================
# Main
# ============================================================


def main():
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200 * 4))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    parser = argparse.ArgumentParser(
        description="ColBERT pretraining with streaming parquet data pipeline"
    )

    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--document_length", type=int, default=305)
    parser.add_argument("--query_length", type=int, default=37)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=110_000,
        help="Total optimizer steps; the streaming dataset is sized to match",
    )
    parser.add_argument("--stop_at_step", type=int, default=90_000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="lightonai/embeddings-pre-training-curated",
    )
    parser.add_argument(
        "--multilingual_dataset_pattern",
        type=str,
        default="",
        help="HF dataset ID pattern with {lang} placeholder, e.g. 'lightonai/mgte-{lang}'",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="ar,de,es,fr,it,no,pt,sv",
        help="Comma-separated language codes for multilingual datasets",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--rg_window", type=int, default=3)
    parser.add_argument("--num_io_threads", type=int, default=4)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (= HF shards)",
    )

    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    model_shortname = args.model_name.split("/")[-1]

    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    temp_str = str(args.temperature).replace(".", "")

    run_name = (
        f"ColBERT_{model_shortname}_"
        f"lr{lr_str}-temp{temp_str}-"
        f"bs{args.batch_size}-mbs{args.mini_batch_size}"
    )

    output_dir = f"./output/pre-training-colbert/{model_shortname}/{run_name}"

    _log(f"\n{'=' * 60}")
    _log("ColBERT Pretraining Configuration:")
    _log(f"{'=' * 60}")
    _log(f"Model: {args.model_name}")
    _log(f"Query Length: {args.query_length} (+5 = {args.query_length + 5})")
    _log(f"Document Length: {args.document_length} (+5 = {args.document_length + 5})")
    _log(f"Learning Rate: {args.learning_rate}")
    _log(f"Temperature: {args.temperature}")
    _log(f"Batch Size: {args.batch_size}")
    _log(f"Mini Batch Size: {args.mini_batch_size}")
    _log(f"Max Steps: {args.max_steps}")
    _log(f"Stop at Step: {args.stop_at_step if args.stop_at_step > 0 else 'Disabled'}")
    _log(f"RG Window: {args.rg_window}")
    _log(f"I/O Threads: {args.num_io_threads}")
    _log(f"DataLoader Workers: {args.num_workers}")
    _log(f"Run Name: {run_name}")
    _log(f"Output Dir: {output_dir}")
    _log(f"{'=' * 60}\n")

    base_model = Transformer(args.model_name)

    # Projection head: in_features must match the backbone hidden size;
    # dense_3 out_features is the per-token embedding dim.
    dense_1 = models.Dense(
        in_features=768,
        out_features=1536,
        bias=False,
        activation_function=torch.nn.Identity(),
        use_residual=True,
    )
    dense_2 = models.Dense(
        in_features=1536,
        out_features=768,
        bias=False,
        activation_function=torch.nn.Identity(),
        use_residual=True,
    )
    dense_3 = models.Dense(
        in_features=768,
        out_features=128,
        bias=False,
        activation_function=torch.nn.Identity(),
        use_residual=False,
    )

    model = models.ColBERT(
        modules=[base_model, dense_1, dense_2, dense_3],
        query_length=args.query_length + 5,
        document_length=args.document_length + 5,
    )

    train_loss = losses.CachedContrastive(
        model,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=True,
        temperature=args.temperature,
    )

    dev_evaluator = MultilingualNanoBEIREvaluator()

    callbacks = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=24,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        fp16=False,
        bf16=True,
        seed=42,
        run_name=run_name,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=2,
        accelerator_config={
            "split_batches": True,
        },
    )

    data_collator = utils.ColBERTCollator(tokenize_fn=model.tokenize)

    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        languages = (
            [
                language.strip()
                for language in args.languages.split(",")
                if language.strip()
            ]
            if args.multilingual_dataset_pattern
            else None
        )
        dataset_file_map = build_dataset_file_map(
            dataset_id=args.dataset_id,
            cache_dir=args.cache_dir,
            multilingual_pattern=args.multilingual_dataset_pattern or None,
            languages=languages,
        )

        ds_config = DatasetConfig(
            dataset_file_map=dataset_file_map,
            batch_size=args.batch_size,
            alpha=0.5,  # dataset sampling: 0 = uniform, 1 = proportional to size
            seed=42,
            steps_per_epoch=args.max_steps,
            rg_window=args.rg_window,
            max_active_cursors=20,
            num_io_threads=args.num_io_threads,
            num_workers=args.num_workers,
        )

        hf_dataset = HFIterableDataset.from_generator(
            _iterate_sharded,
            gen_kwargs=ds_config.gen_kwargs_for_shard(0),
            features=Features({"query": Value("string"), "positive": Value("string")}),
        )
        train_dataset = MultiWorkerDataset(hf_dataset, ds_config)

        _log(
            f"Created dataset with {len(dataset_file_map)} datasets, "
            f"{args.num_workers} DataLoader workers"
        )
        _log(
            f"Pipeline: num_workers={args.num_workers}, prefetch_factor=2, "
            f"rg_window={args.rg_window}, io_threads={args.num_io_threads}"
        )

    accelerator.wait_for_everyone()

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=dev_evaluator,
        loss=train_loss,
        callbacks=callbacks,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}/final")
    _log(f"\n{'=' * 60}")
    _log("Training completed!")
    _log(f"Model saved to: {output_dir}/final")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
