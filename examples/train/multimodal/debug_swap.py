"""Swap internal components between trained and reloaded model to find
exactly which part causes the embedding difference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_swap.py
"""

from __future__ import annotations

import logging
import os
import shutil
import copy

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
OUTPUT_DIR = "models/_debug_swap"
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


def get_emb(model, doc_input, query_input):
    model.eval()
    with torch.no_grad():
        d = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        q = torch.tensor(model.encode([query_input], is_query=True, batch_size=1)[0]).float()
    return d, q


def compare(label, a, b, ref_label="ref"):
    for name, va, vb in [("doc", a[0], b[0]), ("query", a[1], b[1])]:
        delta = (va - vb).abs()
        cos = torch.nn.functional.cosine_similarity(va.flatten().unsqueeze(0), vb.flatten().unsqueeze(0)).item()
        if delta.max().item() == 0:
            log.info("  %s %s: IDENTICAL", label, name)
        else:
            log.info("  %s %s: max=%.2e mean=%.2e cos=%.8f", label, name, delta.max().item(), delta.mean().item(), cos)


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}

    log.info("Training model for 3 steps...")
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
        output_dir=OUTPUT_DIR, run_name="_debug_swap",
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

    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    emb_trained = get_emb(model, doc_input, query_input)
    log.info("Trained model embeddings captured")

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

    emb_reloaded = get_emb(model2, doc_input, query_input)
    log.info("Reloaded model embeddings captured")

    log.info("=" * 70)
    log.info("BASELINE: trained vs reloaded")
    compare("baseline", emb_trained, emb_reloaded)

    # Test 1: Load state_dict from reloaded into trained
    log.info("")
    log.info("=" * 70)
    log.info("TEST 1: Load reloaded state_dict into trained model")
    log.info("=" * 70)
    sd_trained = {k: v.clone() for k, v in model.state_dict().items()}
    sd_reloaded = model2.state_dict()

    # Check if state_dicts have same keys
    only_trained = set(sd_trained) - set(sd_reloaded)
    only_reloaded = set(sd_reloaded) - set(sd_trained)
    if only_trained:
        log.warning("  Keys only in trained: %s", sorted(only_trained)[:5])
    if only_reloaded:
        log.warning("  Keys only in reloaded: %s", sorted(only_reloaded)[:5])

    # Check values
    diffs = []
    for k in sorted(set(sd_trained) & set(sd_reloaded)):
        a, b = sd_trained[k].float(), sd_reloaded[k].float()
        if a.shape == b.shape:
            md = (a - b).abs().max().item()
            if md > 0:
                diffs.append((k, md))
    log.info("  State dict diffs: %d out of %d common keys", len(diffs), len(set(sd_trained) & set(sd_reloaded)))
    for k, md in diffs[:10]:
        log.info("    %s: max_diff=%.2e", k, md)

    # Load reloaded state_dict into trained model
    model.load_state_dict(sd_reloaded, strict=False)
    emb_trained_with_reloaded_sd = get_emb(model, doc_input, query_input)
    compare("trained+reloaded_sd vs reloaded", emb_trained_with_reloaded_sd, emb_reloaded)
    compare("trained+reloaded_sd vs orig_trained", emb_trained_with_reloaded_sd, emb_trained)

    # Restore trained state dict
    model.load_state_dict(sd_trained, strict=False)

    # Test 2: Swap auto_model
    log.info("")
    log.info("=" * 70)
    log.info("TEST 2: Swap auto_model from reloaded into trained")
    log.info("=" * 70)
    orig_auto = model[0].auto_model
    model[0].auto_model = model2[0].auto_model
    emb_swapped_auto = get_emb(model, doc_input, query_input)
    compare("trained+reloaded_auto vs reloaded", emb_swapped_auto, emb_reloaded)
    compare("trained+reloaded_auto vs trained", emb_swapped_auto, emb_trained)
    model[0].auto_model = orig_auto  # restore

    # Test 3: Swap Dense module
    log.info("")
    log.info("=" * 70)
    log.info("TEST 3: Swap Dense module from reloaded into trained")
    log.info("=" * 70)
    orig_dense = model._modules["1"]
    model._modules["1"] = model2._modules["1"]
    emb_swapped_dense = get_emb(model, doc_input, query_input)
    compare("trained+reloaded_dense vs reloaded", emb_swapped_dense, emb_reloaded)
    compare("trained+reloaded_dense vs trained", emb_swapped_dense, emb_trained)
    model._modules["1"] = orig_dense  # restore

    # Test 4: Swap processor
    log.info("")
    log.info("=" * 70)
    log.info("TEST 4: Swap processor from reloaded into trained")
    log.info("=" * 70)
    orig_proc = model[0].processor
    model[0].processor = model2[0].processor
    emb_swapped_proc = get_emb(model, doc_input, query_input)
    compare("trained+reloaded_proc vs reloaded", emb_swapped_proc, emb_reloaded)
    compare("trained+reloaded_proc vs trained", emb_swapped_proc, emb_trained)
    model[0].processor = orig_proc  # restore

    # Test 5: Swap config
    log.info("")
    log.info("=" * 70)
    log.info("TEST 5: Swap config from reloaded into trained auto_model")
    log.info("=" * 70)
    orig_config = model[0].auto_model.config
    model[0].auto_model.config = model2[0].auto_model.config
    emb_swapped_config = get_emb(model, doc_input, query_input)
    compare("trained+reloaded_config vs reloaded", emb_swapped_config, emb_reloaded)
    compare("trained+reloaded_config vs trained", emb_swapped_config, emb_trained)
    model[0].auto_model.config = orig_config  # restore


if __name__ == "__main__":
    main()
