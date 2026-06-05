"""Test autocast configurations on actual retrieval NDCG.

Configs:
  1. bf16 (baseline)
  2. bf16 + autocast(bf16)
  3. bf16 + autocast(fp16)
  4. fp32 + autocast(bf16)  ← training-like setup
  5. fp32 (reference)

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_autocast_eval.py
"""

from __future__ import annotations

import contextlib
import logging
import sys

logging.basicConfig(level=logging.WARNING)

import torch
from datasets import load_dataset
from pylate import evaluation, models

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3_5_4b_colpali_contrastive_simple_native/final"
DOCUMENT_PROMPT = "Describe the image."

EVAL_DATASETS = [
    "vidore/arxivqa_test_subsampled_beir",
    "vidore/infovqa_test_subsampled_beir",
]


def build_evaluators():
    evals = []
    for repo in EVAL_DATASETS:
        queries_ds = load_dataset(repo, "queries", split="test")
        corpus_ds = load_dataset(repo, "corpus", split="test")
        qrels_ds = load_dataset(repo, "qrels", split="test")

        queries = {str(r["query-id"]): r["query"] for r in queries_ds}
        corpus = {str(r["corpus-id"]): {"image": r["image"], "text": DOCUMENT_PROMPT} for r in corpus_ds}
        relevant_docs = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r["query-id"])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r["corpus-id"]))

        short = repo.split("/")[-1].replace("_beir", "")
        evals.append((short, evaluation.PyLateInformationRetrievalEvaluator(
            queries=queries, corpus=corpus, relevant_docs=relevant_docs,
            name=short, batch_size=16, corpus_chunk_size=32, ndcg_at_k=[5],
        )))
    return evals


class AutocastColBERT(models.ColBERT):
    """ColBERT wrapper that runs encode under autocast."""

    _autocast_ctx = None

    def encode(self, *args, **kwargs):
        if self._autocast_ctx is not None:
            with self._autocast_ctx():
                return super().encode(*args, **kwargs)
        return super().encode(*args, **kwargs)


def run_eval(label, model_kwargs, autocast_ctx=None):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    model = AutocastColBERT(model_name_or_path=MODEL_DIR, device="cuda", **model_kwargs)
    model.eval()
    model._autocast_ctx = autocast_ctx
    print(f"  dtype={next(model.parameters()).dtype}  autocast={'yes' if autocast_ctx else 'no'}")

    results = {}
    for short, ev in evaluators:
        with torch.no_grad():
            r = ev(model, output_path=None)
        ndcg = r.get(f"{short}_MaxSim_ndcg@5", 0)
        results[short] = ndcg
        print(f"  {short:45s} ndcg@5={ndcg:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    global evaluators
    print("Loading evaluators...")
    evaluators = build_evaluators()

    all_results = {}

    # 1. bf16 baseline
    all_results["bf16"] = run_eval(
        "BF16 (baseline)", {})

    # 2. bf16 + autocast(bf16)
    all_results["bf16+ac_bf16"] = run_eval(
        "BF16 + autocast(bf16)", {},
        lambda: torch.autocast("cuda", dtype=torch.bfloat16))

    # 3. bf16 + autocast(fp16)
    all_results["bf16+ac_fp16"] = run_eval(
        "BF16 + autocast(fp16)", {},
        lambda: torch.autocast("cuda", dtype=torch.float16))

    # 4. fp32 + autocast(bf16) — training-like setup
    all_results["fp32+ac_bf16"] = run_eval(
        "FP32 + autocast(bf16) [training-like]",
        {"model_kwargs": {"torch_dtype": torch.float32}},
        lambda: torch.autocast("cuda", dtype=torch.bfloat16))

    # 5. fp32 reference
    all_results["fp32"] = run_eval(
        "FP32 (reference)",
        {"model_kwargs": {"torch_dtype": torch.float32}})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Config':<35s}"
    for short, _ in evaluators:
        header += f" {short:>20s}"
    print(header)
    print("-" * 80)
    for cfg, results in all_results.items():
        row = f"{cfg:<35s}"
        for short, _ in evaluators:
            row += f" {results.get(short, 0):>20.4f}"
        print(row)


if __name__ == "__main__":
    main()
