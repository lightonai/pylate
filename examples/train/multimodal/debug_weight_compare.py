"""Compare every parameter between an in-process trained model and its
saved-then-reloaded copy. No eval — just weight comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_weight_compare.py
"""

from __future__ import annotations

import logging
import os
import shutil

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
from peft import LoraConfig, TaskType
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import NoDuplicatesBatchSampler

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "models/_debug_weight_compare"
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
SEED = 42

KD_HUB_DATASET = "lightonai/seer-colpali-queries-v1"
IMAGE_HUB_DATASET = "lightonai/seer-colpali-images-v4"


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


def snapshot_params(model, label=""):
    """Return dict of {name: tensor_clone} for every parameter."""
    params = {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu().clone()
    log.info("[%s] Snapshot: %d params, total elements: %s",
             label, len(params), sum(p.numel() for p in params.values()))
    return params


def compare_params(snap_a, snap_b, label_a="A", label_b="B"):
    """Compare two param snapshots. Report every difference."""
    all_keys = set(snap_a.keys()) | set(snap_b.keys())
    only_a = set(snap_a.keys()) - set(snap_b.keys())
    only_b = set(snap_b.keys()) - set(snap_a.keys())
    common = set(snap_a.keys()) & set(snap_b.keys())

    if only_a:
        log.warning("Keys only in %s (%d):", label_a, len(only_a))
        for k in sorted(only_a)[:10]:
            log.warning("  %s", k)
    if only_b:
        log.warning("Keys only in %s (%d):", label_b, len(only_b))
        for k in sorted(only_b)[:10]:
            log.warning("  %s", k)

    diffs = []
    exact_matches = 0
    for k in sorted(common):
        a, b = snap_a[k].float(), snap_b[k].float()
        if a.shape != b.shape:
            log.error("SHAPE MISMATCH %s: %s vs %s", k, a.shape, b.shape)
            diffs.append((k, float("inf"), float("inf")))
            continue
        delta = (a - b).abs()
        max_diff = delta.max().item()
        mean_diff = delta.mean().item()
        if max_diff > 0:
            diffs.append((k, max_diff, mean_diff))
        else:
            exact_matches += 1

    log.info("=" * 70)
    log.info("COMPARISON: %s vs %s", label_a, label_b)
    log.info("  Common keys: %d", len(common))
    log.info("  Exact matches: %d", exact_matches)
    log.info("  Keys with any diff: %d", len(diffs))

    if diffs:
        diffs.sort(key=lambda x: -x[1])
        log.info("  Top differences (sorted by max_diff):")
        for k, md, mn in diffs[:30]:
            is_lora = "lora_" in k
            is_dense = "Dense" in k or "linear" in k.split(".")[-2:]
            tag = " [LoRA]" if is_lora else (" [Dense]" if is_dense else " [BASE]")
            log.info("    %s%s: max=%.2e mean=%.2e", k, tag, md, mn)

        lora_diffs = [d for d in diffs if "lora_" in d[0]]
        base_diffs = [d for d in diffs if "lora_" not in d[0]]
        log.info("  LoRA param diffs: %d, Base param diffs: %d", len(lora_diffs), len(base_diffs))

        if base_diffs:
            log.warning("  *** BASE MODEL WEIGHTS DIFFER — this should NOT happen ***")
    else:
        log.info("  ALL parameters match exactly!")

    return diffs


def compare_embeddings(model_a, model_b, label_a="A", label_b="B"):
    """Run a forward pass on both models with identical input, compare output."""
    from PIL import Image
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        doc_emb_a = model_a.encode([doc_input], is_query=False, batch_size=1)
        doc_emb_b = model_b.encode([doc_input], is_query=False, batch_size=1)
        query_emb_a = model_a.encode([query_input], is_query=True, batch_size=1)
        query_emb_b = model_b.encode([query_input], is_query=True, batch_size=1)

    for name, emb_a, emb_b in [("doc", doc_emb_a, doc_emb_b), ("query", query_emb_a, query_emb_b)]:
        a = torch.tensor(emb_a[0]).float()
        b = torch.tensor(emb_b[0]).float()
        delta = (a - b).abs()
        max_diff = delta.max().item()
        mean_diff = delta.mean().item()
        cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
        log.info("  %s embeddings: max_diff=%.2e, mean_diff=%.2e, cosine=%.6f, shape=%s",
                 name, max_diff, mean_diff, cos, a.shape)


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}

    # ── Build and train ─────────────────────────────────────────────
    log.info("Loading base model...")
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
        gather_across_devices=False, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_weight_compare",
        num_train_epochs=1, max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, warmup_ratio=0.0,
        bf16=True, fp16=False,
        batch_sampler=NoDuplicatesBatchSampler,
        eval_strategy="no", save_strategy="no",
        logging_steps=1, seed=SEED,
        dataloader_num_workers=2, report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train_datasets,
        loss=loss, data_collator=data_collator,
    )
    trainer.train()

    # ── Snapshot in-process weights ─────────────────────────────────
    log.info("=" * 70)
    log.info("SNAPSHOTTING IN-PROCESS MODEL")
    log.info("=" * 70)
    snap_trained = snapshot_params(model, "trained")

    # Also snapshot module-level state
    log.info("unpad_inputs (trained): %s", model[0].unpad_inputs)
    log.info("model.training (trained): %s", model[0].auto_model.training)

    # ── Save ────────────────────────────────────────────────────────
    save_path = f"{OUTPUT_DIR}/final"
    log.info("Saving to %s", save_path)
    model.save_pretrained(save_path)

    # List what was saved
    for root, dirs, files in os.walk(save_path):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            log.info("  saved: %s (%d bytes)", os.path.relpath(fp, save_path), size)

    # ── Reload ──────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("RELOADING MODEL FROM %s", save_path)
    log.info("=" * 70)

    del trainer, loss
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

    log.info("unpad_inputs (reloaded): %s", model2[0].unpad_inputs)
    log.info("model.training (reloaded): %s", model2[0].auto_model.training)

    snap_reloaded = snapshot_params(model2, "reloaded")

    # ── Compare parameters ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("PARAMETER COMPARISON: trained vs reloaded")
    log.info("=" * 70)
    diffs = compare_params(snap_trained, snap_reloaded, "trained", "reloaded")

    # ── Compare embeddings ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("EMBEDDING COMPARISON: trained vs reloaded")
    log.info("=" * 70)
    model.eval()
    compare_embeddings(model, model2, "trained", "reloaded")

    # ── Also compare base weights against fresh HuggingFace load ────
    log.info("=" * 70)
    log.info("BASE WEIGHT CHECK: reloaded base vs fresh HuggingFace base")
    log.info("=" * 70)
    log.info("Loading fresh base model from HuggingFace for comparison...")
    fresh_base = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    snap_fresh = snapshot_params(fresh_base, "fresh_base")

    # Compare base (non-LoRA, non-Dense) params between trained and fresh
    base_trained = {k: v for k, v in snap_trained.items() if "lora_" not in k and "1_Dense" not in k}
    base_fresh = {k: v for k, v in snap_fresh.items()}
    # The fresh model won't have LoRA keys, so filter to common
    common_base = set(base_trained.keys()) & set(base_fresh.keys())
    log.info("Comparing %d base params (trained vs fresh HuggingFace)...", len(common_base))
    base_diffs = []
    for k in sorted(common_base):
        a, b = base_trained[k].float(), base_fresh[k].float()
        if a.shape != b.shape:
            continue
        delta = (a - b).abs()
        md = delta.max().item()
        if md > 0:
            base_diffs.append((k, md, delta.mean().item()))
    if base_diffs:
        log.warning("BASE weights differ from fresh HuggingFace! %d params changed:", len(base_diffs))
        for k, md, mn in sorted(base_diffs, key=lambda x: -x[1])[:20]:
            log.warning("  %s: max=%.2e mean=%.2e", k, md, mn)
    else:
        log.info("All %d base params match fresh HuggingFace exactly — base is frozen as expected.", len(common_base))


if __name__ == "__main__":
    main()
