"""Train a few steps, eval in-process, save, reload, eval again, compare.

Usage:
    accelerate launch debug_save_reload.py
"""

from __future__ import annotations

import logging
import os
import shutil
from statistics import mean

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

import torch
from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
from peft import LoraConfig, TaskType
from pylate import evaluation, losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import NoDuplicatesBatchSampler
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "models/_debug_save_reload"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784
MAX_PIXELS = 360_000
MAX_SEQ_LENGTH = 4096
DOCUMENT_PROMPT = "Describe the image."

LORA_R = 32
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["down_proj", "gate_proj", "up_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
LORA_EXCLUDE_MODULES = "visual.blocks.*"

TEMPERATURE = 0.02
BATCH_SIZE = 16
MINI_BATCH_SIZE = 4
LEARNING_RATE = 6e-5
MAX_STEPS = 5
EVAL_STEPS = 5
SEED = 42

KD_HUB_DATASET = "lightonai/seer-colpali-queries-v1"
IMAGE_HUB_DATASET = "lightonai/seer-colpali-images-v4"

BENCHMARK_NAME = "vidore_v1_smoke"
EVAL_DATASETS = [
    "vidore/arxivqa_test_subsampled_beir",
    "vidore/infovqa_test_subsampled_beir",
]
EVAL_BATCH_SIZE = 16
EVAL_CORPUS_CHUNK_SIZE = 32
NDCG_K = 5


def _short_name(repo: str) -> str:
    name = repo.split("/", 1)[-1]
    return name[:-len("_beir")] if name.endswith("_beir") else name


class _MacroEvaluator(SequentialEvaluator):
    def __init__(self, sub_evaluators, *, name, primary_ndcg_k):
        super().__init__(sub_evaluators, main_score_function=mean)
        self.name = name
        self._macro_key = f"{name}_macro_MaxSim_ndcg@{primary_ndcg_k}"
        self.primary_metric = self._macro_key

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path, epoch, steps)
        results[self._macro_key] = results.pop("sequential_score")
        self.primary_metric = self._macro_key
        return results


def build_evaluator():
    sub_evaluators = []
    for repo in EVAL_DATASETS:
        queries_ds = load_dataset(repo, "queries", split="test")
        corpus_ds = load_dataset(repo, "corpus", split="test")
        qrels_ds = load_dataset(repo, "qrels", split="test")
        queries = {str(r["query-id"]): r["query"] for r in queries_ds}
        corpus = {str(r["corpus-id"]): r["image"] for r in corpus_ds}
        relevant_docs = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r["query-id"])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r["corpus-id"]))
        sub_evaluators.append(
            evaluation.PyLateInformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=relevant_docs,
                name=f"{BENCHMARK_NAME}_{_short_name(repo)}",
                batch_size=EVAL_BATCH_SIZE, corpus_chunk_size=EVAL_CORPUS_CHUNK_SIZE,
                ndcg_at_k=[NDCG_K],
            )
        )
    return _MacroEvaluator(sub_evaluators, name=BENCHMARK_NAME, primary_ndcg_k=NDCG_K)


def load_one_split():
    images_ds = load_dataset(IMAGE_HUB_DATASET, split="train")
    images_ds = images_ds.cast_column("image", DatasetImage(decode=False))
    filenames = images_ds.with_format(None)["image_filename"]
    image_index = {name: i for i, name in enumerate(filenames)}

    split = "arxiv_qa"
    queries_ds = load_dataset(KD_HUB_DATASET, "queries", split=split)
    documents_ds = load_dataset(KD_HUB_DATASET, "documents", split=split)
    scores_ds = load_dataset(KD_HUB_DATASET, "scores", split=split)

    query_index = {qid: i for i, qid in enumerate(queries_ds["query_id"])}
    doc_index = {did: i for i, did in enumerate(documents_ds["document_id"])}
    all_queries = queries_ds["query"]
    all_doc_filenames = documents_ds["image_filename"]
    all_score_qids = scores_ds["query_id"]
    all_score_doc_ids = scores_ds["document_ids"]

    rows = {"query": [], "positive_image_idx": [], "negative_image_idx": []}
    for i in range(len(scores_ds)):
        doc_ids = all_score_doc_ids[i]
        if len(doc_ids) < 2:
            continue
        query_text = all_queries[query_index[all_score_qids[i]]]
        pos_fn = all_doc_filenames[doc_index[doc_ids[0]]]
        neg_fn = all_doc_filenames[doc_index[doc_ids[1]]]
        if pos_fn not in image_index or neg_fn not in image_index:
            continue
        rows["query"].append(query_text)
        rows["positive_image_idx"].append(image_index[pos_fn])
        rows["negative_image_idx"].append(image_index[neg_fn])

    ds = Dataset.from_dict(rows)
    decode_image = DatasetImage(decode=True)

    def transform(batch):
        out = {"query": batch["query"], "image": [], "negative_0": []}
        for pos_idx, neg_idx in zip(batch["positive_image_idx"], batch["negative_image_idx"]):
            pos_pil = decode_image.decode_example(images_ds[pos_idx]["image"])
            neg_pil = decode_image.decode_example(images_ds[neg_idx]["image"])
            out["image"].append({"image": pos_pil, "text": DOCUMENT_PROMPT})
            out["negative_0"].append({"image": neg_pil, "text": DOCUMENT_PROMPT})
        return out

    ds.set_transform(transform)
    return ds, images_ds


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model[0].unpad_inputs = False

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES, init_lora_weights=True,
        use_rslora=True, use_dora=False,
    )
    model.add_adapter(lora_config)

    train_ds, _ = load_one_split()
    train_datasets = DatasetDict({"arxiv_qa": train_ds})

    loss = losses.CachedContrastive(
        model=model, mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=True, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_save_reload",
        num_train_epochs=1, max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, warmup_ratio=0.0,
        bf16=True, fp16=False,
        batch_sampler=NoDuplicatesBatchSampler,
        accelerator_config={"split_batches": True},
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=MAX_STEPS,
        save_total_limit=1, logging_steps=1, seed=SEED,
        dataloader_num_workers=2, report_to=[],
    )

    evaluator = build_evaluator()

    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train_datasets,
        loss=loss, data_collator=data_collator, evaluator=evaluator,
    )
    trainer.train()

    # ── Post-training eval (in-process, same model object) ──────────
    print("=" * 60)
    print("IN-PROCESS EVAL (same model object, after training)")
    print("=" * 60)
    results_in_process = evaluator(model)
    for k, v in sorted(results_in_process.items()):
        if "ndcg" in k:
            print(f"  IN-PROCESS {k}: {v:.4f}")

    # ── Save ────────────────────────────────────────────────────────
    save_path = f"{OUTPUT_DIR}/final"
    print(f"Saving to {save_path}")
    model.save_pretrained(save_path)

    # ── Reload and eval ─────────────────────────────────────────────
    print("=" * 60)
    print(f"RELOADING model from {save_path}")
    print("=" * 60)
    del model, trainer, loss
    torch.cuda.empty_cache()

    model2 = models.ColBERT(
        model_name_or_path=save_path,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model2[0].unpad_inputs = False
    model2.eval()

    print("=" * 60)
    print("RELOADED EVAL")
    print("=" * 60)
    results_reloaded = evaluator(model2)
    for k, v in sorted(results_reloaded.items()):
        if "ndcg" in k:
            print(f"  RELOADED  {k}: {v:.4f}")

    # ── Compare ─────────────────────────────────────────────────────
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for k in sorted(results_in_process):
        if "ndcg" in k:
            v1, v2 = results_in_process[k], results_reloaded[k]
            print(f"  {k}: in_process={v1:.4f} reloaded={v2:.4f} diff={v2 - v1:+.4f}")


if __name__ == "__main__":
    main()
