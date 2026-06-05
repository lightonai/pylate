"""Check if torch.compile or persistent autocast explains the forward diff.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_autocast_compile.py
"""

from __future__ import annotations

import logging
import os
import shutil
from functools import wraps

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
OUTPUT_DIR = "models/_debug_autocast"
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


def check_autocast_and_compile(model, label):
    log.info("=" * 70)
    log.info("STATE CHECK: %s", label)
    log.info("=" * 70)
    log.info("  torch.is_autocast_enabled(): %s", torch.is_autocast_enabled())
    log.info("  torch.is_autocast_enabled('cuda'): %s", torch.is_autocast_enabled('cuda'))
    log.info("  torch.get_autocast_dtype('cuda'): %s", torch.get_autocast_dtype('cuda'))

    auto_model = model[0].model
    log.info("  type(auto_model): %s", type(auto_model).__name__)
    log.info("  auto_model.__class__: %s", auto_model.__class__)
    log.info("  hasattr _compiled: %s", hasattr(auto_model, '_compiled'))
    log.info("  hasattr _dynamo: %s", hasattr(auto_model, '_dynamo'))
    log.info("  hasattr _torchdynamo_orig_callable: %s", hasattr(auto_model, '_torchdynamo_orig_callable'))

    # Check for torch.compile wrapping
    import types
    fwd = auto_model.forward
    log.info("  forward type: %s", type(fwd).__name__)
    log.info("  forward module: %s", getattr(fwd, '__module__', 'N/A'))
    if hasattr(fwd, '__wrapped__'):
        log.info("  forward.__wrapped__: %s", fwd.__wrapped__)

    # Check for dynamo
    if hasattr(auto_model, '_dynamo_ctx'):
        log.info("  _dynamo_ctx: %s", auto_model._dynamo_ctx)

    # Check if any submodule is compiled
    compiled_modules = []
    for name, mod in auto_model.named_modules():
        if hasattr(mod, '_compiled') or isinstance(mod, torch._dynamo.OptimizedModule if hasattr(torch._dynamo, 'OptimizedModule') else type(None)):
            compiled_modules.append(name)
    if compiled_modules:
        log.warning("  Compiled modules: %s", compiled_modules[:5])
    else:
        log.info("  No compiled modules found")

    # Check model._parameters vs state_dict for hidden state
    sd = auto_model.state_dict()
    params = dict(auto_model.named_parameters())
    bufs = dict(auto_model.named_buffers())
    sd_keys = set(sd.keys())
    param_buf_keys = set(params.keys()) | set(bufs.keys())
    extra_in_sd = sd_keys - param_buf_keys
    if extra_in_sd:
        log.warning("  Extra keys in state_dict (not params/buffers): %s", sorted(extra_in_sd)[:10])
    else:
        log.info("  state_dict matches params+buffers exactly (%d keys)", len(sd_keys))


def get_emb(model, doc_input, query_input):
    model.eval()
    with torch.no_grad():
        d = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        q = torch.tensor(model.encode([query_input], is_query=True, batch_size=1)[0]).float()
    return d, q


def compare(label, a, b):
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

    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    log.info("BEFORE TRAINING:")
    check_autocast_and_compile(model, "before_training")
    emb_before = get_emb(model, doc_input, query_input)

    # Train
    train_ds, _ = load_one_split()
    train_datasets = DatasetDict({"arxiv_qa": train_ds})

    loss = losses.CachedContrastive(
        model=model, mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=False, temperature=TEMPERATURE,
    )
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, run_name="_debug_autocast",
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

    log.info("")
    log.info("AFTER TRAINING:")
    check_autocast_and_compile(model, "after_training")
    emb_after = get_emb(model, doc_input, query_input)

    # Key test: encode with explicit autocast disabled
    log.info("")
    log.info("ENCODE WITH EXPLICIT AUTOCAST DISABLED:")
    with torch.autocast("cuda", enabled=False):
        emb_no_autocast = get_emb(model, doc_input, query_input)

    # Encode with explicit autocast enabled
    log.info("ENCODE WITH EXPLICIT AUTOCAST BF16:")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        emb_with_autocast = get_emb(model, doc_input, query_input)

    log.info("")
    log.info("=" * 70)
    log.info("COMPARISONS")
    log.info("=" * 70)
    compare("after_training vs no_autocast", emb_after, emb_no_autocast)
    compare("after_training vs with_autocast", emb_after, emb_with_autocast)
    compare("no_autocast vs with_autocast", emb_no_autocast, emb_with_autocast)

    # Save and reload
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

    log.info("")
    log.info("RELOADED:")
    check_autocast_and_compile(model2, "reloaded")
    emb_reloaded = get_emb(model2, doc_input, query_input)

    log.info("")
    log.info("=" * 70)
    log.info("FINAL COMPARISON")
    log.info("=" * 70)
    compare("trained vs reloaded", emb_after, emb_reloaded)
    compare("trained_no_autocast vs reloaded", emb_no_autocast, emb_reloaded)
    compare("trained_with_autocast vs reloaded", emb_with_autocast, emb_reloaded)


if __name__ == "__main__":
    main()
