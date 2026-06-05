"""ViDoRe v2 + v3 eval with autocast(bf16) for correct scores.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_vidore_v2_v3.py [model_path]
"""

from __future__ import annotations

import logging
import sys
from statistics import mean

logging.basicConfig(level=logging.WARNING)

import torch
from datasets import load_dataset
from pylate import evaluation, models
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3_5_4b_colpali_contrastive_simple_native/final"
DOCUMENT_PROMPT = "Describe the image."
NDCG_K = 5

V2_DATASETS = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_v2",
    "vidore/economics_reports_v2",
]

V3_DATASETS = [
    "vidore/vidore_v3_finance_en",
    "vidore/vidore_v3_hr",
    "vidore/vidore_v3_industrial",
    "vidore/vidore_v3_pharmaceuticals",
    "vidore/vidore_v3_computer_science",
    "vidore/vidore_v3_energy",
    "vidore/vidore_v3_physics",
]

EVAL_BATCH_SIZE = 16
EVAL_CORPUS_CHUNK_SIZE = 32


def short_name(repo: str) -> str:
    return repo.split("/")[-1]


def load_beir_dataset(repo: str):
    """Load a BEIR-format dataset, handling both v2 (hyphenated) and v3 (underscored) key names."""
    queries_ds = load_dataset(repo, "queries", split="test")
    corpus_ds = load_dataset(repo, "corpus", split="test")
    qrels_ds = load_dataset(repo, "qrels", split="test")

    qid_col = "query-id" if "query-id" in queries_ds.column_names else "query_id"
    cid_col = "corpus-id" if "corpus-id" in corpus_ds.column_names else "corpus_id"
    qrel_qid = "query-id" if "query-id" in qrels_ds.column_names else "query_id"
    qrel_cid = "corpus-id" if "corpus-id" in qrels_ds.column_names else "corpus_id"

    queries = {str(r[qid_col]): r["query"] for r in queries_ds}
    corpus = {
        str(r[cid_col]): {"image": r["image"], "text": DOCUMENT_PROMPT}
        for r in corpus_ds
    }
    relevant_docs: dict[str, set[str]] = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qid = str(r[qrel_qid])
            if qid in queries:
                relevant_docs.setdefault(qid, set()).add(str(r[qrel_cid]))

    return queries, corpus, relevant_docs


class _MacroEvaluator(SequentialEvaluator):
    def __init__(self, sub_evaluators, *, name: str) -> None:
        super().__init__(sub_evaluators, main_score_function=mean)
        self.name = name
        self._macro_key = f"{name}_macro_MaxSim_ndcg@{NDCG_K}"
        self.primary_metric = self._macro_key

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path, epoch, steps)
        results[self._macro_key] = results.pop("sequential_score")
        self.primary_metric = self._macro_key
        return results


class AutocastColBERT(models.ColBERT):
    def encode(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return super().encode(*args, **kwargs)


def build_evaluator(datasets, benchmark_name):
    sub_evaluators = []
    for repo in datasets:
        name = short_name(repo)
        print(f"  Loading {repo}...")
        queries, corpus, relevant_docs = load_beir_dataset(repo)
        print(f"    {len(queries)} queries, {len(corpus)} docs, {sum(len(v) for v in relevant_docs.values())} qrels")
        sub_evaluators.append(
            evaluation.PyLateInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"{benchmark_name}_{name}",
                batch_size=EVAL_BATCH_SIZE,
                corpus_chunk_size=EVAL_CORPUS_CHUNK_SIZE,
                ndcg_at_k=[NDCG_K],
            )
        )
    return _MacroEvaluator(sub_evaluators, name=benchmark_name)


def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Model: {MODEL_DIR}")
    print(f"Loading model with autocast(bf16)...")
    model = AutocastColBERT(model_name_or_path=MODEL_DIR, device="cuda")
    model.eval()
    print(f"  dtype={next(model.parameters()).dtype}")

    for benchmark_name, datasets in [("vidore_v2", V2_DATASETS), ("vidore_v3", V3_DATASETS)]:
        output_dir = f"{MODEL_DIR.rstrip('/')}/../eval_{benchmark_name}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 72}")
        print(f"Building {benchmark_name} evaluator ({len(datasets)} datasets)...")
        evaluator = build_evaluator(datasets, benchmark_name)

        print(f"Running {benchmark_name} eval...")
        with torch.no_grad():
            results = evaluator(model, output_path=output_dir)

        print(f"\n{benchmark_name} results (ndcg@{NDCG_K}, MaxSim, autocast bf16):")
        for repo in datasets:
            name = short_name(repo)
            key = f"{benchmark_name}_{name}_MaxSim_ndcg@{NDCG_K}"
            if key in results:
                print(f"  {name:<45s} {results[key]:.4f}")
        macro_key = f"{benchmark_name}_macro_MaxSim_ndcg@{NDCG_K}"
        if macro_key in results:
            print(f"  {'MACRO AVERAGE':<45s} {results[macro_key]:.4f}")


if __name__ == "__main__":
    main()
