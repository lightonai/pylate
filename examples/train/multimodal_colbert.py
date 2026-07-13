"""Train a ColPali-style late-interaction retriever on Qwen3.5-4B.

The model is trained with LoRA + contrastive loss on two KD-mined datasets:
  - lightonai/colpali-train-fine-tuning + lightonai/colpali-train-images
    (ViDoRe v1/v2 domains: arxiv_qa, tatdqa, docvqa, pdf, infographic_vqa)
  - lightonai/llamaindex-vdr-fine-tuning + lightonai/llamaindex-vdr-images
    (VDR-multilingual: de, en, es, fr, it)
Both fine-tuning datasets share the same layout: a "queries" config
(query_id -> text), a "documents" config (document_id -> image_filename),
and a "scores" config where each row holds a query_id plus document_ids
ranked by a teacher score (first document = positive, the rest are negative
candidates). Images live in a separate dataset keyed by image_filename.

For each query we keep the positive image plus up to MAX_NEGATIVES hard
negatives whose teacher score is below NV_THRESHOLD * positive_score (to
avoid false negatives), then sample SAMPLE_NEGATIVES of them per batch at
transform time.

Usage:
    accelerate launch --mixed_precision=bf16 colpali_qwen_3_5_vidore_v1v2_vdr_v2_clean.py
"""

from __future__ import annotations

import io
import logging
import os
import random

from accelerate import PartialState
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)
from datasets import (
    Image as DatasetImage,
)
from peft import LoraConfig, TaskType
from PIL import Image as PIL_Image
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import NoDuplicatesBatchSampler
from transformers import AutoModelForImageTextToText

from pylate import evaluation, losses, models, utils

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── Run config ───────────────────────────────────────────────────────────────
# "native": use the processor's built-in chat template.
# "colpali": use the minimal ColPali-style template defined below.
CHAT_TEMPLATE_VARIANT = "native"

RUN_NAME = (
    f"qwen3_5_4b_colpali_contrastive_vidore_v1v2_vdr_v2_{CHAT_TEMPLATE_VARIANT}_3e-5"
)
OUTPUT_DIR = f"models/{RUN_NAME}"
WANDB_PROJECT = "multimodal_pylate"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_3"
EMBEDDING_SIZE = 128  # ColBERT projection dim
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 3136  # 4 x 28x28 patches minimum
MAX_PIXELS = 1_003_520  # match ColNomic
MAX_SEQ_LENGTH = 4096

# ── Chat template ────────────────────────────────────────────────────────────
# Text paired with every document image, so documents render as
# "<vision tokens> + prompt" instead of vision tokens alone. Whether it must
# be passed explicitly depends on the chat template variant:
#   - "native": the processor's built-in template adds no text to an
#     image-only message, so the prompt MUST be attached to each document —
#     in the train transform ({"image": ..., "text": DOCUMENT_PROMPT}) and
#     via the evaluator's document_prompt argument.
#   - "colpali": the custom template below already falls back to
#     DOCUMENT_PROMPT when the message carries no text item, so documents
#     could be passed as bare images (drop the "text" key in the transform
#     and set document_prompt=None in the evaluator) with identical results.
# We always attach it explicitly so both variants see the exact same inputs;
# whatever you choose, train and eval must be consistent.
DOCUMENT_PROMPT = "Describe the image."

# Minimal ColPali-style template: images render as
# "<|im_start|>user\n<vision tokens>{prompt}<|im_end|><|endoftext|>",
# text-only inputs (queries) render as the raw text with no chat wrapping.
QWEN_VL_COLPALI_CHAT_TEMPLATE = (
    "{%- set msg = messages[0] -%}"
    "{%- set ns = namespace(has_image=false, text='') -%}"
    "{%- for item in msg.content -%}"
    "{%- if item.type == 'image' -%}{%- set ns.has_image = true -%}"
    "{%- elif item.type == 'text' -%}{%- set ns.text = item.text -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.has_image -%}"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{{ ns.text if ns.text else '" + DOCUMENT_PROMPT + "' }}"
    "<|im_end|><|endoftext|>"
    "{%- else -%}"
    "{{ ns.text }}"
    "{%- endif -%}"
)

CHAT_TEMPLATE = (
    QWEN_VL_COLPALI_CHAT_TEMPLATE if CHAT_TEMPLATE_VARIANT == "colpali" else None
)

# ── LoRA config ──────────────────────────────────────────────────────────────
# Qwen3.5 mixes full self-attention layers (q/k/v/o_proj) with linear-attention
# layers (in_proj_qkv/in_proj_z/out_proj), so both sets must be targeted to
# cover all 32 LM layers. The vision encoder blocks are frozen, but the
# vision->LM merger (linear_fc1/fc2) is adapted.
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    # LM self-attention layers
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # LM linear-attention layers
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    # LM MLP (all layers)
    "down_proj",
    "gate_proj",
    "up_proj",
    # Vision merger (vision-to-LM bridge)
    "linear_fc1",
    "linear_fc2",
]
LORA_EXCLUDE_MODULES = "visual.blocks.*"  # freeze vision encoder; merger stays adapted

# ── Trainer / loss config ────────────────────────────────────────────────────
TEMPERATURE = 0.02
BATCH_SIZE = 32
MINI_BATCH_SIZE = 4  # CachedContrastive gradient-cache chunk size
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.005
MAX_STEPS = 3125
SEED = 42
EVAL_STEPS = 250
DDP_TIMEOUT = 14400  # 4h — rank-0 ViDoRe eval on the 4B backbone blocks other ranks

# ── Data: ColPali (ViDoRe) ───────────────────────────────────────────────────
KD_HUB_DATASET = "lightonai/colpali-train-fine-tuning"
IMAGE_HUB_DATASET = "lightonai/colpali-train-images"
SPLITS = ["arxiv_qa", "tatdqa", "docvqa", "pdf", "infographic_vqa"]

# ── Data: VDR-multilingual ───────────────────────────────────────────────────
VDR_KD_DATASET = "lightonai/llamaindex-vdr-fine-tuning"
VDR_IMAGE_HUB_DATASET = "lightonai/llamaindex-vdr-images"
VDR_LANGUAGES = ["de", "en", "es", "fr", "it"]

# ── Negative filtering / sampling ────────────────────────────────────────────
NV_THRESHOLD = 0.95  # keep negatives with score < threshold * positive_score
MAX_NEGATIVES = 20  # max negatives stored per query
MIN_NEGATIVES = 1  # queries with fewer valid negatives are dropped
SAMPLE_NEGATIVES = 3  # negatives sampled per query at transform time

# ── Eval config ──────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 16
EVAL_CORPUS_CHUNK_SIZE = 32


# ── Model fixups ─────────────────────────────────────────────────────────────
# I fix it has been fixed in newer version but letting it there just in case
def fix_qwen35_backbone_weights(model: models.ColBERT) -> None:
    """Reload the LM tower via AutoModelForImageTextToText if it loaded broken.

    Some transformers versions load the Qwen3.5 multimodal checkpoint with the
    language_model weights left uninitialised (all-zero layernorms). Detect
    that and swap in a correctly loaded copy. Must run BEFORE adding LoRA.
    """
    transformer = model[0]
    probe = transformer.model.language_model.layers[0].input_layernorm.weight
    if float(probe.abs().sum().item()) != 0.0:
        log.warning(
            "Qwen3.5 LM tower loaded correctly (layernorm sum != 0) — no fix needed."
        )
        return

    log.warning(
        "Qwen3.5 LM tower loaded BROKEN (layernorm sum == 0) — reloading via "
        "AutoModelForImageTextToText."
    )
    wrapper = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
    )
    target_device = next(transformer.model.parameters()).device
    transformer.model = wrapper.to(target_device).model
    del wrapper

    probe = transformer.model.language_model.layers[0].input_layernorm.weight
    if float(probe.abs().sum().item()) == 0.0:
        raise RuntimeError(
            "Qwen3.5 language_model weights still uninitialised after the "
            "AutoModelForImageTextToText reload — the multimodal loading path "
            "has regressed. Inspect Qwen3_5Model.base_model_prefix."
        )
    log.info("Applied Qwen3.5 LM-tower fix (reloaded via AutoModelForImageTextToText).")


def verify_chat_template(model: models.ColBERT) -> None:
    """Render a document (image + prompt) and a text-only query and log both."""
    processor = model[0].processor
    kwargs = getattr(model[0], "processing_kwargs", {}).get("chat_template", {})
    dummy = PIL_Image.new("RGB", (28, 28), color="white")

    doc_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy},
                {"type": "text", "text": DOCUMENT_PROMPT},
            ],
        }
    ]
    query_messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is shown here?"}]}
    ]

    doc_render = processor.apply_chat_template(doc_messages, tokenize=False, **kwargs)
    query_render = processor.apply_chat_template(
        query_messages, tokenize=False, **kwargs
    )
    log.info(
        "Chat-template render check (variant=%r):\n  DOCUMENT:\n%s\n  QUERY:\n%s",
        CHAT_TEMPLATE_VARIANT,
        doc_render,
        query_render,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def build_image_index(images_ds: Dataset) -> dict[str, int]:
    """Map image_filename -> row index for fast lookup."""
    filenames = images_ds.with_format(None)["image_filename"]
    return {name: i for i, name in enumerate(filenames)}


def load_seer_images() -> Dataset:
    """Load the seer-colpali image dataset from the Hub, without decoding."""
    log.info("Loading image dataset %s...", IMAGE_HUB_DATASET)
    images_ds = load_dataset(IMAGE_HUB_DATASET, split="train")
    # Keep raw bytes; PIL decoding happens lazily in the batch transform.
    return images_ds.cast_column("image", DatasetImage(decode=False))


def load_vdr_images() -> Dataset:
    """Load the VDR image dataset from the Hub, without decoding."""
    log.info("Loading VDR image dataset %s...", VDR_IMAGE_HUB_DATASET)
    images_ds = load_dataset(VDR_IMAGE_HUB_DATASET, split="train")
    return images_ds.cast_column("image", DatasetImage(decode=False))


def _filter_negatives(
    doc_ids: list[int],
    score_values: list[float],
    doc_index: dict[int, int],
    all_doc_filenames: list[str],
    image_index: dict[str, int],
) -> list[int]:
    """Return image indices for negatives passing the nv-threshold filter.

    doc_ids[0] is the positive; negatives scoring too close to it (likely
    false negatives) are discarded.
    """
    threshold = NV_THRESHOLD * score_values[0]
    valid = []
    for j in range(1, len(doc_ids)):
        if score_values[j] < threshold:
            fname = all_doc_filenames[doc_index[doc_ids[j]]]
            if fname in image_index:
                valid.append(image_index[fname])
                if len(valid) >= MAX_NEGATIVES:
                    break
    return valid


def load_kd_split(hub_dataset: str, split: str, image_index: dict[str, int]) -> Dataset:
    """Load one KD split and filter negatives.

    Produces rows of (query text, positive image index, candidate negative
    image indices). Queries missing their positive image or with fewer than
    MIN_NEGATIVES valid negatives are dropped.
    """
    log.info("Loading split %s/%s...", hub_dataset, split)
    queries_ds = load_dataset(hub_dataset, "queries", split=split)
    documents_ds = load_dataset(hub_dataset, "documents", split=split)
    scores_ds = load_dataset(hub_dataset, "scores", split=split)

    query_index = {qid: i for i, qid in enumerate(queries_ds["query_id"])}
    doc_index = {did: i for i, did in enumerate(documents_ds["document_id"])}

    all_queries = queries_ds["query"]
    all_doc_filenames = documents_ds["image_filename"]
    all_score_qids = scores_ds["query_id"]
    all_score_doc_ids = scores_ds["document_ids"]
    all_scores = scores_ds["scores"]

    rows = {"query": [], "positive_image_idx": [], "negative_image_idxs": []}
    skipped = 0
    for i in range(len(scores_ds)):
        doc_ids = all_score_doc_ids[i]
        score_values = all_scores[i]
        if len(doc_ids) < 2:
            skipped += 1
            continue

        pos_filename = all_doc_filenames[doc_index[doc_ids[0]]]
        if pos_filename not in image_index:
            skipped += 1
            continue

        valid_negs = _filter_negatives(
            doc_ids,
            score_values,
            doc_index,
            all_doc_filenames,
            image_index,
        )
        if len(valid_negs) < MIN_NEGATIVES:
            skipped += 1
            continue

        rows["query"].append(all_queries[query_index[all_score_qids[i]]])
        rows["positive_image_idx"].append(image_index[pos_filename])
        rows["negative_image_idxs"].append(valid_negs)

    log.info(
        "Split %s: %d rows kept, %d skipped (nv_threshold=%.2f).",
        split,
        len(rows["query"]),
        skipped,
        NV_THRESHOLD,
    )
    return Dataset.from_dict(rows)


def make_image_transform(images_ds: Dataset):
    """Build a set_transform callback that decodes images and samples negatives.

    Rows only store image indices; here we decode the positive image and
    SAMPLE_NEGATIVES randomly sampled negatives to PIL, pairing each with the
    document prompt. Undecodable negatives are skipped; if fewer than
    SAMPLE_NEGATIVES decode, the decoded ones are repeated to fill the slots.
    """

    def decode(idx: int) -> PIL_Image.Image:
        image_bytes = images_ds[int(idx)]["image"]["bytes"]
        return PIL_Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def transform(batch):
        out = {"query": batch["query"], "image": []}
        for n in range(SAMPLE_NEGATIVES):
            out[f"negative_{n}"] = []

        for i in range(len(batch["query"])):
            pos_pil = decode(batch["positive_image_idx"][i])
            out["image"].append({"image": pos_pil, "text": DOCUMENT_PROMPT})

            neg_pool = batch["negative_image_idxs"][i]
            decoded_negs = []
            for candidate_idx in random.sample(neg_pool, len(neg_pool)):
                if len(decoded_negs) >= SAMPLE_NEGATIVES:
                    break
                try:
                    decoded_negs.append(decode(candidate_idx))
                except Exception:
                    continue
            for n in range(SAMPLE_NEGATIVES):
                neg_pil = decoded_negs[n % len(decoded_negs)]
                out[f"negative_{n}"].append({"image": neg_pil, "text": DOCUMENT_PROMPT})
        return out

    return transform


def prepare_splits(
    state: PartialState,
    cache_dir: str,
    split_names: dict[str, str],
    load_images_fn,
    hub_dataset: str,
) -> dict[str, Dataset]:
    """Build (rank 0) and load (all ranks) the filtered splits for one source.

    Rank 0 loads the raw KD data, filters negatives, and caches the resulting
    lightweight splits to disk; the other ranks wait, then every rank
    memory-maps the cached splits and attaches the image-decoding transform.

    split_names maps the cache/train-dataset name to the Hub split name.
    """
    if state.is_main_process:
        images_ds = load_images_fn()
        image_index = build_image_index(images_ds)
        os.makedirs(cache_dir, exist_ok=True)
        for name, hub_split in split_names.items():
            path = os.path.join(cache_dir, name)
            if not os.path.exists(path):
                log.warning("Building split cache for %s...", name)
                ds = load_kd_split(hub_dataset, hub_split, image_index)
                ds.save_to_disk(path)
                log.warning("Cached %s (%d rows).", name, len(ds))
        del image_index
    state.wait_for_everyone()
    if not state.is_main_process:
        images_ds = load_images_fn()

    transform = make_image_transform(images_ds)
    datasets = {}
    for name in split_names:
        ds = load_from_disk(os.path.join(cache_dir, name))
        ds.set_transform(transform)
        datasets[name] = ds
    return datasets


def main() -> None:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # ── 1. Model ─────────────────────────────────────────────────────────────
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "model_max_length": MAX_SEQ_LENGTH,
    }
    if CHAT_TEMPLATE is not None:
        processor_kwargs["chat_template"] = {"chat_template": CHAT_TEMPLATE}

    log.info(
        "Loading model %s (chat template: %s)...", MODEL_NAME, CHAT_TEMPLATE_VARIANT
    )
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={
            "attn_implementation": ATTN_IMPLEMENTATION,
            "torch_dtype": TORCH_DTYPE,
        },
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="",
        document_prefix="",
        query_length=QUERY_LENGTH,
        document_length=DOCUMENT_LENGTH,
        do_query_expansion=True,
        attend_to_expansion_tokens=True,
    )

    # Fixups must happen before LoRA wraps the modules.
    fix_qwen35_backbone_weights(model)
    model[0].unpad_inputs = False
    verify_chat_template(model)
    log.info("Model loaded.")

    # ── 2. LoRA ──────────────────────────────────────────────────────────────
    log.info("Adding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES,
    )
    model.add_adapter(lora_config)

    # ── 3. Data ──────────────────────────────────────────────────────────────
    state = PartialState()

    train_datasets = prepare_splits(
        state,
        cache_dir=os.path.join(OUTPUT_DIR, ".splits_cache"),
        split_names={name: name for name in SPLITS},
        load_images_fn=load_seer_images,
        hub_dataset=KD_HUB_DATASET,
    )
    train_datasets.update(
        prepare_splits(
            state,
            cache_dir=os.path.join(OUTPUT_DIR, ".vdr_splits_cache"),
            split_names={f"vdr_{lang}": f"train_{lang}" for lang in VDR_LANGUAGES},
            load_images_fn=load_vdr_images,
            hub_dataset=VDR_KD_DATASET,
        )
    )
    train_datasets = DatasetDict(train_datasets)
    log.info(
        "All training splits loaded: %s",
        {name: len(ds) for name, ds in train_datasets.items()},
    )

    # ── 4. Loss ──────────────────────────────────────────────────────────────
    # CachedContrastive = in-batch contrastive with gradient caching, so the
    # effective batch (gathered across devices) fits in memory via mini-batches.
    loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=True,
        temperature=TEMPERATURE,
    )

    # ── 5. Trainer ───────────────────────────────────────────────────────────
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        bf16=True,
        fp16=False,
        batch_sampler=NoDuplicatesBatchSampler,
        accelerator_config={"split_batches": True},
        multi_dataset_batch_sampler="proportional",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=1,
        seed=SEED,
        dataloader_num_workers=8,
        ddp_timeout=DDP_TIMEOUT,
        report_to=["wandb"],
    )

    # ── 6. Evaluator (ViDoRe v2) ─────────────────────────────────────────────
    log.info("Building ViDoRe evaluator...")
    evaluator = evaluation.ViDoREvaluator(
        dataset_names=["esgreports"],
        # Must match training: with the native template nothing adds a prompt
        # to image-only documents, so it has to be passed here too (see the
        # DOCUMENT_PROMPT comment at the top).
        document_prompt=DOCUMENT_PROMPT,
        corpus_chunk_size=EVAL_CORPUS_CHUNK_SIZE,
        batch_size=EVAL_BATCH_SIZE,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        loss=loss,
        data_collator=data_collator,
        evaluator=evaluator,
    )
    log.info("Starting training...")
    trainer.train()

    # ── 8. Final eval + save ─────────────────────────────────────────────────
    log.info("Running post-training evaluation...")
    evaluator(model)
    log.info("Saving model to %s...", f"{OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    log.info("Done.")


if __name__ == "__main__":
    main()
