"""Full ViDoRe v1 eval in fp32 to check if precision recovers training-time scores.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_vidore_full_qwen35_fp32.py
    # or point at a different checkpoint:
    CUDA_VISIBLE_DEVICES=0 python eval_vidore_full_qwen35_fp32.py /path/to/model/dir
"""

from __future__ import annotations

import logging
import sys
from statistics import mean

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

import torch
from datasets import load_dataset
from pylate import evaluation, models
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

DEFAULT_MODEL_DIR = (
    "models/qwen3_5_4b_colpali_contrastive_simple_native/final"
)
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
EVAL_OUTPUT_DIR = f"{MODEL_DIR.rstrip('/')}/../eval_vidore_v1_fp32"

BENCHMARK_NAME = "vidore_v1"
EVAL_DATASETS = [
    "vidore/arxivqa_test_subsampled_beir",
    "vidore/docvqa_test_subsampled_beir",
    "vidore/infovqa_test_subsampled_beir",
    "vidore/tabfquad_test_subsampled_beir",
    "vidore/tatdqa_test_beir",
    "vidore/shiftproject_test_beir",
    "vidore/syntheticDocQA_artificial_intelligence_test_beir",
    "vidore/syntheticDocQA_energy_test_beir",
    "vidore/syntheticDocQA_government_reports_test_beir",
    "vidore/syntheticDocQA_healthcare_industry_test_beir",
]
EVAL_BATCH_SIZE = 16
EVAL_CORPUS_CHUNK_SIZE = 32
NDCG_K = 5
DOCUMENT_PROMPT = "Describe the image."


def _short_dataset_name(repo: str) -> str:
    name = repo.split("/", 1)[-1]
    return name[: -len("_beir")] if name.endswith("_beir") else name


class _MacroEvaluator(SequentialEvaluator):
    def __init__(self, sub_evaluators, *, name: str, primary_ndcg_k: int) -> None:
        super().__init__(sub_evaluators, main_score_function=mean)
        self.name = name
        self._macro_key = f"{name}_macro_MaxSim_ndcg@{primary_ndcg_k}"
        self.primary_metric = self._macro_key

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path, epoch, steps)
        results[self._macro_key] = results.pop("sequential_score")
        self.primary_metric = self._macro_key
        return results


def build_vidore_evaluator() -> SequentialEvaluator:
    sub_evaluators = []
    for repo in EVAL_DATASETS:
        print(f"Loading eval dataset {repo}...")
        queries_ds = load_dataset(repo, "queries", split="test")
        corpus_ds = load_dataset(repo, "corpus", split="test")
        qrels_ds = load_dataset(repo, "qrels", split="test")

        queries = {str(r["query-id"]): r["query"] for r in queries_ds}
        corpus = {
            str(r["corpus-id"]): {"image": r["image"], "text": DOCUMENT_PROMPT}
            for r in corpus_ds
        }
        relevant_docs: dict[str, set[str]] = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r["query-id"])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r["corpus-id"]))

        sub_name = f"{BENCHMARK_NAME}_{_short_dataset_name(repo)}"
        sub_evaluators.append(
            evaluation.PyLateInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=sub_name,
                batch_size=EVAL_BATCH_SIZE,
                corpus_chunk_size=EVAL_CORPUS_CHUNK_SIZE,
                ndcg_at_k=[NDCG_K],
            )
        )

    return _MacroEvaluator(sub_evaluators, name=BENCHMARK_NAME, primary_ndcg_k=NDCG_K)


def main() -> None:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_DIR} in FP32...")
    model = models.ColBERT(
        model_name_or_path=MODEL_DIR,
        model_kwargs={"torch_dtype": torch.float32},
        device="cuda",
    )
    model.eval()

    actual_dtype = next(model.parameters()).dtype
    print(f"Model dtype after load: {actual_dtype}")

    if actual_dtype != torch.float32:
        print("Casting model to fp32...")
        model.float()
        actual_dtype = next(model.parameters()).dtype
        print(f"Model dtype after cast: {actual_dtype}")

    evaluator = build_vidore_evaluator()

    print(f"Running ViDoRe v1 eval in FP32 (output -> {EVAL_OUTPUT_DIR})...")
    with torch.no_grad():
        results = evaluator(model, output_path=EVAL_OUTPUT_DIR)

    print("=" * 72)
    print(f"ViDoRe v1 results (ndcg@{NDCG_K}, MaxSim, FP32):")
    per_dataset = {}
    for repo in EVAL_DATASETS:
        short = _short_dataset_name(repo)
        key = f"{BENCHMARK_NAME}_{short}_MaxSim_ndcg@{NDCG_K}"
        if key in results:
            per_dataset[short] = results[key]
            print(f"  {short:<48s} {results[key]:.4f}")
    macro_key = f"{BENCHMARK_NAME}_macro_MaxSim_ndcg@{NDCG_K}"
    if macro_key in results:
        print(f"  {'MACRO AVERAGE':<48s} {results[macro_key]:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
