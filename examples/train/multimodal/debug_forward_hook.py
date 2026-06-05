"""Hook into model forward calls to compare exact inputs and outputs
between a trained model and its reloaded copy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_forward_hook.py
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
OUTPUT_DIR = "models/_debug_forward_hook"
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


class ForwardCapture:
    """Register a pre-forward hook on the auto_model to capture kwargs."""
    def __init__(self, model, label):
        self.label = label
        self.captured_kwargs = {}
        self.captured_output = None
        auto_model = model[0].auto_model
        self._handle = auto_model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        self.captured_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self.captured_kwargs[k] = v.detach().cpu().clone()
            else:
                self.captured_kwargs[k] = v
        return args, kwargs

    def remove(self):
        self._handle.remove()


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}

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
        output_dir=OUTPUT_DIR, run_name="_debug_fwd_hook",
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

    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}

    # Check model_forward_params
    log.info("=" * 70)
    log.info("MODEL FORWARD PARAMS")
    log.info("=" * 70)
    mfp_a = model[0].model_forward_params
    mfp_b = model2[0].model_forward_params
    log.info("  Trained: %s", sorted(mfp_a))
    log.info("  Reloaded: %s", sorted(mfp_b))
    if mfp_a != mfp_b:
        log.warning("  *** DIFFER: only in trained=%s, only in reloaded=%s",
                     mfp_a - mfp_b, mfp_b - mfp_a)

    # Check modality_config
    log.info("  Trained modality_config: %s", model[0].modality_config)
    log.info("  Reloaded modality_config: %s", model2[0].modality_config)

    # Check auto_model config differences
    log.info("=" * 70)
    log.info("AUTO_MODEL CONFIG COMPARISON")
    log.info("=" * 70)
    cfg_a = model[0].auto_model.config
    cfg_b = model2[0].auto_model.config
    for attr in dir(cfg_a):
        if attr.startswith("_"):
            continue
        va = getattr(cfg_a, attr, None)
        vb = getattr(cfg_b, attr, None)
        if callable(va):
            continue
        if va != vb:
            log.warning("  CONFIG DIFF: %s = %s vs %s", attr, va, vb)

    # Hook into forward and capture kwargs
    log.info("=" * 70)
    log.info("FORWARD CAPTURE: doc encoding")
    log.info("=" * 70)

    cap_a = ForwardCapture(model, "trained")
    model.eval()
    with torch.no_grad():
        emb_a = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()
    cap_a.remove()

    cap_b = ForwardCapture(model2, "reloaded")
    with torch.no_grad():
        emb_b = torch.tensor(model2.encode([doc_input], is_query=False, batch_size=1)[0]).float()
    cap_b.remove()

    # Compare captured kwargs
    all_keys = set(cap_a.captured_kwargs) | set(cap_b.captured_kwargs)
    log.info("  Captured kwargs keys (trained): %s", sorted(cap_a.captured_kwargs.keys()))
    log.info("  Captured kwargs keys (reloaded): %s", sorted(cap_b.captured_kwargs.keys()))

    for k in sorted(all_keys):
        va = cap_a.captured_kwargs.get(k, "MISSING")
        vb = cap_b.captured_kwargs.get(k, "MISSING")
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                log.warning("  KWARG %s SHAPE: %s vs %s", k, va.shape, vb.shape)
            else:
                diff = (va.float() - vb.float()).abs()
                if diff.max().item() > 0:
                    log.warning("  KWARG %s TENSOR DIFF: max=%.2e mean=%.2e", k, diff.max().item(), diff.mean().item())
                else:
                    log.info("  KWARG %s: identical tensor (shape=%s, dtype=%s)", k, va.shape, va.dtype)
        elif va != vb:
            log.warning("  KWARG %s: trained=%s, reloaded=%s", k, va, vb)
        else:
            log.info("  KWARG %s: identical (%s)", k, type(va).__name__)

    # Compare embeddings
    delta = (emb_a - emb_b).abs()
    cos = torch.nn.functional.cosine_similarity(emb_a.flatten().unsqueeze(0), emb_b.flatten().unsqueeze(0)).item()
    log.info("  EMBEDDINGS: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f", delta.max().item(), delta.mean().item(), cos)


if __name__ == "__main__":
    main()
