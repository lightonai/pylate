"""Check parameter dtypes after training vs after reload.

Hypothesis: mixed-precision training upcasts LoRA params to fp32 for optimizer
precision, but save/reload stores/loads them in bf16 -> the forward pass computes
differently (fp32 matmul vs bf16 matmul).

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_dtype_check.py
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import Counter

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
OUTPUT_DIR = "models/_debug_dtype_check"
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


def report_dtypes(model, label):
    log.info("=" * 70)
    log.info("DTYPE REPORT: %s", label)
    log.info("=" * 70)

    lora_dtypes = Counter()
    base_dtypes = Counter()
    dense_dtypes = Counter()

    for name, p in model.named_parameters():
        if "lora_" in name:
            lora_dtypes[str(p.dtype)] += 1
        elif "1_Dense" in name or (name.endswith(".linear.weight") or name.endswith(".linear.bias")):
            dense_dtypes[str(p.dtype)] += 1
        else:
            base_dtypes[str(p.dtype)] += 1

    log.info("  LoRA params: %s", dict(lora_dtypes))
    log.info("  Dense params: %s", dict(dense_dtypes))
    log.info("  Base params: %s", dict(base_dtypes))

    # Show a few specific LoRA params
    for name, p in model.named_parameters():
        if "lora_" in name and "layers.0." in name:
            log.info("  Example: %s dtype=%s device=%s", name, p.dtype, p.device)


def main():
    from PIL import Image

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}

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

    report_dtypes(model, "BEFORE TRAINING")

    train_ds, _ = load_one_split()
    train_datasets = DatasetDict({"arxiv_qa": train_ds})

    loss = losses.CachedContrastive(
        model=model, mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=False, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_dtype",
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

    report_dtypes(model, "AFTER TRAINING (in-process)")

    # Get embedding from trained model
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    model.eval()
    with torch.no_grad():
        emb_trained = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    # Now manually cast LoRA params to bf16 (simulating what reload does)
    log.info("=" * 70)
    log.info("CASTING LoRA params from fp32 -> bf16 (simulating reload dtype)")
    log.info("=" * 70)
    cast_count = 0
    for name, p in model.named_parameters():
        if "lora_" in name and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            cast_count += 1
    log.info("  Cast %d LoRA params to bf16", cast_count)

    report_dtypes(model, "AFTER MANUAL CAST TO BF16")

    with torch.no_grad():
        emb_after_cast = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta = (emb_trained - emb_after_cast).abs()
    cos = torch.nn.functional.cosine_similarity(
        emb_trained.flatten().unsqueeze(0), emb_after_cast.flatten().unsqueeze(0)).item()
    log.info("=" * 70)
    log.info("EMBEDDING DIFF: fp32-LoRA vs bf16-LoRA (same values, different dtype)")
    log.info("  max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta.max().item(), delta.mean().item(), cos)

    # Also cast Dense layer
    for name, p in model.named_parameters():
        if p.dtype == torch.float32:
            log.info("  Still fp32: %s", name)
            p.data = p.data.to(torch.bfloat16)

    report_dtypes(model, "ALL PARAMS IN BF16")

    with torch.no_grad():
        emb_all_bf16 = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta2 = (emb_trained - emb_all_bf16).abs()
    cos2 = torch.nn.functional.cosine_similarity(
        emb_trained.flatten().unsqueeze(0), emb_all_bf16.flatten().unsqueeze(0)).item()
    log.info("EMBEDDING DIFF: original fp32 vs all-bf16")
    log.info("  max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta2.max().item(), delta2.mean().item(), cos2)

    # Save, reload, compare
    log.info("=" * 70)
    log.info("SAVE/RELOAD CHECK")
    log.info("=" * 70)
    # Reload the original trained model (before we cast it)
    # Actually we already cast it, so let's just check the checkpoint dtypes
    save_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(save_path)

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

    report_dtypes(model2, "RELOADED MODEL")

    # Also check the REAL checkpoint
    log.info("=" * 70)
    log.info("CHECKING REAL CHECKPOINT DTYPES")
    log.info("=" * 70)
    import safetensors.torch as st
    real_ckpt = "models/qwen3_5_4b_colpali_contrastive_simple_native/checkpoint-3125/adapter_model.safetensors"
    if os.path.exists(real_ckpt):
        saved = st.load_file(real_ckpt)
        dtype_counts = Counter()
        for k, v in saved.items():
            dtype_counts[str(v.dtype)] += 1
        log.info("  Real checkpoint adapter dtypes: %s", dict(dtype_counts))
        for k, v in list(saved.items())[:3]:
            log.info("  Example: %s dtype=%s shape=%s", k, v.dtype, v.shape)


if __name__ == "__main__":
    main()
