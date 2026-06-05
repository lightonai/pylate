"""Compare preprocessing (tokenization + image processing) between a model
loaded from base+LoRA and one loaded from a saved checkpoint.

If the tokenized inputs differ, the embedding difference comes from
preprocessing, not from the model forward pass.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_preprocessing.py
"""

from __future__ import annotations

import logging
import os
import shutil

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from PIL import Image
from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
from peft import LoraConfig, TaskType
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import NoDuplicatesBatchSampler

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "models/_debug_preprocess"
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
MAX_STEPS = 3
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


def compare_preprocessing(model_a, model_b, label_a, label_b):
    log.info("=" * 70)
    log.info("PREPROCESSING COMPARISON: %s vs %s", label_a, label_b)
    log.info("=" * 70)

    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    # Check processor configuration
    proc_a = model_a[0].processor
    proc_b = model_b[0].processor
    log.info("  Processor A type: %s", type(proc_a).__name__)
    log.info("  Processor B type: %s", type(proc_b).__name__)

    # Check processing_kwargs
    pkw_a = getattr(model_a[0], "processing_kwargs", {})
    pkw_b = getattr(model_b[0], "processing_kwargs", {})
    log.info("  Processing kwargs A: %s", pkw_a)
    log.info("  Processing kwargs B: %s", pkw_b)
    if pkw_a != pkw_b:
        log.warning("  *** PROCESSING KWARGS DIFFER ***")
        for k in set(list(pkw_a.keys()) + list(pkw_b.keys())):
            va = pkw_a.get(k, "MISSING")
            vb = pkw_b.get(k, "MISSING")
            if va != vb:
                log.warning("    %s: A=%s, B=%s", k, va, vb)

    # Check chat template
    ct_a = getattr(proc_a, "chat_template", None)
    ct_b = getattr(proc_b, "chat_template", None)
    if ct_a != ct_b:
        log.warning("  *** CHAT TEMPLATE DIFFERS ***")
        log.info("  Chat template A (first 200): %s", str(ct_a)[:200] if ct_a else "None")
        log.info("  Chat template B (first 200): %s", str(ct_b)[:200] if ct_b else "None")
    else:
        log.info("  Chat templates match (both %s chars)", len(ct_a) if ct_a else "None")

    # Check image processor config
    ip_a = getattr(proc_a, "image_processor", None)
    ip_b = getattr(proc_b, "image_processor", None)
    if ip_a and ip_b:
        for attr in ["min_pixels", "max_pixels", "size", "crop_size", "do_resize", "do_normalize",
                      "image_mean", "image_std", "do_rescale", "rescale_factor",
                      "patch_size", "merge_size", "temporal_patch_size"]:
            va = getattr(ip_a, attr, "N/A")
            vb = getattr(ip_b, attr, "N/A")
            if va != vb:
                log.warning("  IMAGE PROC %s: A=%s, B=%s", attr, va, vb)

    # Now use the model's preprocess method to see what tensors are produced
    log.info("")
    log.info("  Using model.preprocess() to compare tensor inputs...")
    preproc_a = model_a.preprocess
    preproc_b = model_b.preprocess

    # Doc preprocessing
    doc_features_a = preproc_a({"image": [doc_input]})
    doc_features_b = preproc_b({"image": [doc_input]})

    log.info("  DOC features A keys: %s", list(doc_features_a.keys()))
    log.info("  DOC features B keys: %s", list(doc_features_b.keys()))

    for k in doc_features_a:
        if k in doc_features_b:
            va = doc_features_a[k]
            vb = doc_features_b[k]
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                if va.shape != vb.shape:
                    log.warning("  DOC %s SHAPE: A=%s, B=%s", k, va.shape, vb.shape)
                else:
                    diff = (va.float() - vb.float()).abs()
                    if diff.max().item() > 0:
                        log.warning("  DOC %s DIFF: max=%.2e mean=%.2e", k, diff.max().item(), diff.mean().item())
                    else:
                        log.info("  DOC %s: identical (shape=%s)", k, va.shape)
            elif va != vb:
                log.warning("  DOC %s: A=%s, B=%s", k, va, vb)

    # Query preprocessing
    query_features_a = preproc_a({"query": [query_input]})
    query_features_b = preproc_b({"query": [query_input]})

    log.info("  QUERY features A keys: %s", list(query_features_a.keys()))
    log.info("  QUERY features B keys: %s", list(query_features_b.keys()))

    for k in query_features_a:
        if k in query_features_b:
            va = query_features_a[k]
            vb = query_features_b[k]
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                if va.shape != vb.shape:
                    log.warning("  QUERY %s SHAPE: A=%s, B=%s", k, va.shape, vb.shape)
                else:
                    diff = (va.float() - vb.float()).abs()
                    if diff.max().item() > 0:
                        log.warning("  QUERY %s DIFF: max=%.2e mean=%.2e", k, diff.max().item(), diff.mean().item())
                    else:
                        log.info("  QUERY %s: identical (shape=%s)", k, va.shape)


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}

    # Build and train model
    log.info("Loading base model and training 3 steps...")
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
        output_dir=OUTPUT_DIR, run_name="_debug_preprocess",
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

    save_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(save_path)

    # Reload
    log.info("Reloading model from %s...", save_path)
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

    compare_preprocessing(model, model2, "trained", "reloaded")

    # Also check the REAL checkpoint
    log.info("")
    log.info("=" * 70)
    log.info("ALSO COMPARING WITH REAL CHECKPOINT")
    log.info("=" * 70)
    real_ckpt = "models/qwen3_5_4b_colpali_contrastive_simple_native/checkpoint-3125"
    if os.path.exists(real_ckpt):
        model3 = models.ColBERT(
            model_name_or_path=real_ckpt,
            model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
            processor_kwargs=processor_kwargs,
            trust_remote_code=True,
            embedding_size=EMBEDDING_SIZE,
            query_prefix="", document_prefix="",
            query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
            do_query_expansion=True, attend_to_expansion_tokens=True,
        )
        model3[0].unpad_inputs = False
        model3.eval()

        compare_preprocessing(model, model3, "trained_debug", "real_checkpoint")

        # Compare real checkpoint vs base model (no adapter)
        log.info("")
        log.info("=" * 70)
        log.info("COMPARING BASE MODEL (no adapter) vs CHECKPOINT")
        log.info("=" * 70)
        model_base = models.ColBERT(
            model_name_or_path=MODEL_NAME,
            model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
            processor_kwargs=processor_kwargs,
            trust_remote_code=True,
            embedding_size=EMBEDDING_SIZE,
            query_prefix="", document_prefix="",
            query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
            do_query_expansion=True, attend_to_expansion_tokens=True,
        )
        compare_preprocessing(model_base, model3, "base_no_adapter", "real_checkpoint")


if __name__ == "__main__":
    main()
