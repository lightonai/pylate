"""Find the hidden state left by training on the model object.

Strategy: compare every attribute on every submodule between
the trained model and the fresh+copy model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_find_hidden_state.py
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
OUTPUT_DIR = "models/_debug_find_state"
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


def compare_module_attrs(mod_t, mod_f, path, diffs):
    """Compare all attributes of two modules."""
    attrs_t = set(vars(mod_t).keys())
    attrs_f = set(vars(mod_f).keys())

    only_t = attrs_t - attrs_f
    only_f = attrs_f - attrs_t
    if only_t:
        for a in sorted(only_t):
            if not a.startswith("__"):
                diffs.append((path, a, "ONLY_IN_TRAINED", getattr(mod_t, a, None)))
    if only_f:
        for a in sorted(only_f):
            if not a.startswith("__"):
                diffs.append((path, a, "ONLY_IN_FRESH", getattr(mod_f, a, None)))

    for attr in sorted(attrs_t & attrs_f):
        if attr.startswith("__"):
            continue
        vt = getattr(mod_t, attr)
        vf = getattr(mod_f, attr)

        # Skip submodules — they'll be compared recursively
        if isinstance(vt, torch.nn.Module) or isinstance(vf, torch.nn.Module):
            continue
        # Skip dict-like containers of modules
        if attr in ("_modules", "_parameters", "_buffers"):
            continue
        # Skip hook dicts — already checked
        if "hook" in attr:
            continue

        if isinstance(vt, torch.Tensor) and isinstance(vf, torch.Tensor):
            if vt.shape != vf.shape:
                diffs.append((path, attr, "SHAPE", f"{vt.shape} vs {vf.shape}"))
            elif not torch.equal(vt, vf):
                md = (vt.float() - vf.float()).abs().max().item()
                diffs.append((path, attr, "TENSOR_DIFF", f"max={md:.2e}"))
        elif isinstance(vt, (bool, int, float, str, type(None))):
            if vt != vf:
                diffs.append((path, attr, "VALUE", f"{vt} vs {vf}"))
        elif isinstance(vt, (list, tuple)):
            if str(vt) != str(vf):
                diffs.append((path, attr, "COLLECTION", f"len {len(vt)} vs {len(vf)}"))
        elif isinstance(vt, dict):
            if set(vt.keys()) != set(vf.keys()):
                diffs.append((path, attr, "DICT_KEYS", f"{set(vt.keys()) - set(vf.keys())} / {set(vf.keys()) - set(vt.keys())}"))


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
        gather_across_devices=False, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_find_state",
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

    # Create fresh model with LoRA and copy weights
    trained_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    del trainer, loss
    torch.cuda.empty_cache()

    fresh = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    fresh[0].unpad_inputs = False
    fresh.add_adapter(LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES, init_lora_weights=True,
        use_rslora=True, use_dora=False,
    ))
    fresh.load_state_dict(trained_sd, strict=False)

    # Deep compare every module attribute
    log.info("=" * 70)
    log.info("DEEP MODULE COMPARISON: trained vs fresh+copy")
    log.info("=" * 70)

    diffs = []
    modules_t = dict(model.named_modules())
    modules_f = dict(fresh.named_modules())

    for name in sorted(set(modules_t.keys()) & set(modules_f.keys())):
        compare_module_attrs(modules_t[name], modules_f[name], name, diffs)

    if diffs:
        log.info("Found %d differences:", len(diffs))
        for path, attr, kind, detail in diffs:
            log.warning("  [%s] %s.%s: %s", kind, path or "(root)", attr, str(detail)[:200])
    else:
        log.info("No differences found in any module attributes!")


if __name__ == "__main__":
    main()
