"""Check all registered hooks and hidden state on the model after training.
Also check if Accelerator wrapping persists.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_hooks_state.py
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
OUTPUT_DIR = "models/_debug_hooks"
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


def count_hooks(model, label):
    log.info("=" * 70)
    log.info("HOOKS ON MODEL: %s", label)
    log.info("=" * 70)
    total_fwd = 0
    total_bwd = 0
    total_pre = 0
    total_state = 0
    for name, module in model.named_modules():
        n_fwd = len(module._forward_hooks)
        n_bwd = len(module._backward_hooks)
        n_pre = len(module._forward_pre_hooks)
        n_state = len(getattr(module, "_state_dict_hooks", {}))
        if n_fwd or n_bwd or n_pre or n_state:
            log.info("  %s: fwd=%d, bwd=%d, pre=%d, state_dict=%d",
                     name or "(root)", n_fwd, n_bwd, n_pre, n_state)
            # Show hook details
            for hid, hook in module._forward_hooks.items():
                log.info("    fwd hook %s: %s", hid, hook)
            for hid, hook in module._forward_pre_hooks.items():
                log.info("    pre hook %s: %s", hid, hook)
            for hid, hook in module._backward_hooks.items():
                log.info("    bwd hook %s: %s", hid, hook)
        total_fwd += n_fwd
        total_bwd += n_bwd
        total_pre += n_pre
        total_state += n_state
    log.info("  Total: fwd=%d, bwd=%d, pre=%d, state_dict=%d", total_fwd, total_bwd, total_pre, total_state)


def check_model_internals(model, label):
    log.info("=" * 70)
    log.info("MODEL INTERNALS: %s", label)
    log.info("=" * 70)

    # Check non-param, non-buffer tensor attributes on a few modules
    auto_model = model[0].auto_model
    for name, module in auto_model.named_modules():
        if "layers.0.self_attn" in name and not any(c in name for c in ["lora_", "."]) or name.endswith("layers.0.self_attn"):
            log.info("  Module: %s (%s)", name, type(module).__name__)
            for attr_name in sorted(vars(module)):
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.Tensor) and attr_name not in dict(module.named_parameters()) and attr_name not in dict(module.named_buffers()):
                    log.info("    Non-param tensor: %s shape=%s dtype=%s", attr_name, attr.shape, attr.dtype)
                elif not attr_name.startswith("_") and not callable(attr) and not isinstance(attr, (torch.nn.Module, dict, list, set)):
                    if attr_name not in ("training",):
                        log.info("    Attr: %s = %s", attr_name, attr)
            break

    # Check the Transformer module itself
    transformer = model[0]
    log.info("  Transformer module type: %s", type(transformer).__name__)
    log.info("  auto_model type: %s", type(auto_model).__name__)
    log.info("  auto_model.__class__.__module__: %s", type(auto_model).__module__)

    # Check for Accelerator wrapping
    log.info("  Has _original_forward: %s", hasattr(auto_model, "_original_forward"))
    log.info("  Has _accelerate_hook: %s", hasattr(auto_model, "_accelerate_hook"))
    for name, module in auto_model.named_modules():
        if hasattr(module, "_hf_hook"):
            log.info("  Module %s has _hf_hook: %s", name, module._hf_hook)
            break

    # Check if gradient checkpointing is enabled
    log.info("  gradient_checkpointing: %s", getattr(auto_model, "gradient_checkpointing", "NOT_SET"))
    log.info("  is_gradient_checkpointing: %s", getattr(auto_model, "is_gradient_checkpointing", "NOT_SET"))

    # Check the Dense module
    dense = model[1]
    log.info("  Dense module: %s", type(dense).__name__)
    log.info("  Dense weight dtype: %s", dense.linear.weight.dtype)


def main():
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

    log.info("BEFORE TRAINING:")
    count_hooks(model, "before_training")
    check_model_internals(model, "before_training")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES, init_lora_weights=True,
        use_rslora=True, use_dora=False,
    )
    model.add_adapter(lora_config)

    log.info("AFTER ADDING LORA:")
    count_hooks(model, "after_lora")

    train_ds, _ = load_one_split()
    train_datasets = DatasetDict({"arxiv_qa": train_ds})

    loss = losses.CachedContrastive(
        model=model, mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=False, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_hooks",
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

    log.info("AFTER TRAINING:")
    count_hooks(model, "after_training")
    check_model_internals(model, "after_training")

    # Compare embedding before and after removing all hooks
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}

    model.eval()
    with torch.no_grad():
        emb_with_hooks = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    # Remove ALL hooks from ALL modules
    log.info("Removing all hooks...")
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
        module._forward_pre_hooks.clear()

    count_hooks(model, "after_removing_hooks")

    with torch.no_grad():
        emb_no_hooks = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta = (emb_with_hooks - emb_no_hooks).abs()
    cos = torch.nn.functional.cosine_similarity(emb_with_hooks.flatten().unsqueeze(0), emb_no_hooks.flatten().unsqueeze(0)).item()
    log.info("EMBEDDING DIFF (with hooks vs no hooks): max=%.2e mean=%.2e cos=%.8f",
             delta.max().item(), delta.mean().item(), cos)

    # Now compare with reloaded
    save_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(save_path)

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

    log.info("RELOADED MODEL:")
    count_hooks(model2, "reloaded")
    check_model_internals(model2, "reloaded")

    with torch.no_grad():
        emb_reloaded = torch.tensor(model2.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta2 = (emb_no_hooks - emb_reloaded).abs()
    cos2 = torch.nn.functional.cosine_similarity(emb_no_hooks.flatten().unsqueeze(0), emb_reloaded.flatten().unsqueeze(0)).item()
    log.info("EMBEDDING DIFF (trained-no-hooks vs reloaded): max=%.2e mean=%.2e cos=%.8f",
             delta2.max().item(), delta2.mean().item(), cos2)


if __name__ == "__main__":
    main()
