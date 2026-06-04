"""Minimal ColPali-style training for BidirLM-Omni-2.5B-Embedding via PyLate.

Self-contained counterpart to ``colpali_simple_explicit_template.py``, but
configured for the omnimodal BidirLM-Omni backbone. Reproduces:
    accelerate launch --mixed_precision=bf16 scripts/train/colbert_contrastive.py \
        --config configs/train/bidirlm_omni_2_5b_colpali_contrastive_v2.yaml

Key differences vs the Qwen2.5-VL script:
  * Model: BidirLM/BidirLM-Omni-2.5B-Embedding (trust_remote_code).
  * SLOWNESS FIX (critical): the vision patch embed uses an nn.Conv3d with
    kernel == stride; with tens of thousands of flattened patches cuDNN picks a
    catastrophic algorithm (~3s/image, >100x slowdown). We replace its forward
    with an equivalent F.linear (~114x faster, bf16-identical). See
    ``patch_bidirlm_omni_patch_embed`` below (ported from seer.models).
  * PIXEL BUDGET: BidirLMOmniProcessor drops max_pixels through the
    processor_kwargs chain, so we re-apply min/max pixels on the image processor
    after construction (``enforce_pixel_budget``, ported from seer.models).
  * Chat template: BidirLM-Omni's NATIVE template adds NO system prompt (unlike
    Qwen2.5-VL-Instruct). We forge that native rendering explicitly as the
    default variant; "colpali" is available for A/B.
  * LoRA also adapts the vision mergers (linear_fc1/linear_fc2) and excludes the
    frozen vision blocks + audio tower.
  * mini_batch_size=4, eval_steps=200, ddp_timeout=14400 (per the YAML).

Usage:
    accelerate launch --mixed_precision=bf16 scripts/train/colpali_simple_bidirlm.py
"""

from __future__ import annotations

import logging
import os
import types
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from datasets import (
    Dataset,
    DatasetDict,
    Image as DatasetImage,
    load_dataset,
)
from peft import LoraConfig, TaskType
from pylate import evaluation, losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sampler import NoDuplicatesBatchSampler
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

# ── Run config ───────────────────────────────────────────────────────────────
# "native"  — BidirLM-Omni's actual chat template (no system prompt). This is
#             what seer training already uses (model_type "bidirlm_omni" is not
#             in PyLate's COLPALI_CHAT_TEMPLATES, so the processor template stands).
# "colpali" — ColPali-faithful (raw query, doc terminated with <|endoftext|>).
CHAT_TEMPLATE_VARIANT = "native"  # "native" | "colpali"

RUN_NAME = f"bidirlm_omni_2_5b_colpali_contrastive_simple_{CHAT_TEMPLATE_VARIANT}"
OUTPUT_DIR = f"models/{RUN_NAME}"
WANDB_PROJECT = "multimodal_pylate"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "BidirLM/BidirLM-Omni-2.5B-Embedding"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784  # 28 * 28
MAX_PIXELS = 300_000
MAX_SEQ_LENGTH = 4096

# ── Chat template ──────────────────────────────────────────────────────────--
# Document instruction text. Seer feeds documents this prompt; the colpali
# template uses it as the fallback for a raw-PIL image with no text item.
DOCUMENT_PROMPT = "Describe the image."
#
# "native"  — DO NOT FORGE. Use BidirLM-Omni's own shipped chat template. Its
#   no-tools / no-system-message branch renders bare text with no system prompt
#   (verified against chat_template.jinja), which is how the model is meant to
#   be used here:
#       query:    <|im_start|>user\n{query}<|im_end|>\n
#       document: <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{doc text}<|im_end|>\n
#   (A system message would switch text to "Instruct: ...\nQuery: ..." — we
#   intentionally pass none, so we ride with the bare-text rendering.)
#
# "colpali" — ColPali-faithful override (raw query, doc terminated with
#   <|endoftext|>, DOCUMENT_PROMPT fallback for raw-PIL docs). A/B baseline.
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

# Only "colpali" forges a template; "native" leaves the shipped one in place.
CHAT_TEMPLATE = QWEN_VL_COLPALI_CHAT_TEMPLATE if CHAT_TEMPLATE_VARIANT == "colpali" else None

# ── LoRA config ──────────────────────────────────────────────────────────────
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# LM uses Qwen3-style *_proj; vision mergers expose linear_fc1/linear_fc2.
LORA_TARGET_MODULES = [
    "down_proj",
    "gate_proj",
    "up_proj",
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "linear_fc1",
    "linear_fc2",
]
# Keep frozen: the vision transformer blocks and the audio encoder layers.
LORA_EXCLUDE_MODULES = r"(visual\.blocks|audio_tower\.layers)\..*"

# ── Loss config ──────────────────────────────────────────────────────────────
TEMPERATURE = 0.02

# ── Trainer config ───────────────────────────────────────────────────────────
BATCH_SIZE = 64
MINI_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.005
MAX_STEPS = 3125
SEED = 42
EVAL_STEPS = 200
DDP_TIMEOUT = 14400  # 4h — rank-0 ViDoRe eval on the 2.5B backbone.

# ── Data paths ───────────────────────────────────────────────────────────────
# KD metadata (queries / documents / scores) now lives on the Hub as three
# configs, each with one split per task — same schema as the old NFS dump at
# /mnt/nfs/seer/data/seer-colpali-kd-metadata-v1.
KD_HUB_DATASET = "lightonai/seer-colpali-queries-v1"
IMAGE_HUB_DATASET = "lightonai/seer-colpali-images-v4"

SPLITS = ["arxiv_qa", "tatdqa", "docvqa", "pdf", "infographic_vqa"]

# ── Eval config ──────────────────────────────────────────────────────────────
BENCHMARK_NAME = "vidore_v1_smoke"
EVAL_DATASETS = [
    "vidore/arxivqa_test_subsampled_beir",
    "vidore/infovqa_test_subsampled_beir",
]
EVAL_BATCH_SIZE = 16
EVAL_CORPUS_CHUNK_SIZE = 32
NDCG_K = 5


# ── Model fixups (ported from seer.models) ──────────────────────────────────--
def patch_bidirlm_omni_patch_embed(model: models.ColBERT) -> None:
    """Replace BidirLM-Omni's Conv3d patch embed with an equivalent F.linear.

    ``BidirLMOmniVisionPatchEmbed`` uses nn.Conv3d with kernel == stride. With a
    flattened-patch input (N_patches, in_channels, T, H, W) and N in the tens of
    thousands, cuDNN picks a catastrophic algorithm — patch embedding alone runs
    at ~0.7 GFLOPs/s on H100 and dominates the vision forward (>3s/image, >100x
    slowdown). Mathematically the kernel is just a per-patch linear projection,
    so we swap forward for an F.linear against the reshaped Conv3d weight.
    ~114x speedup; output matches to bf16 precision.
    """
    transformer = model[0] if len(model) else None
    backbone = getattr(transformer, "model", None)
    visual = getattr(backbone, "visual", None)
    patch_embed = getattr(visual, "patch_embed", None)
    proj = getattr(patch_embed, "proj", None)
    if proj is None or not isinstance(proj, torch.nn.Conv3d):
        log.warning("BidirLM patch-embed fix skipped (no Conv3d proj found).")
        return
    if tuple(proj.stride) != tuple(proj.kernel_size):
        log.warning("BidirLM patch-embed fix skipped (stride != kernel).")
        return

    weight = proj.weight.view(proj.out_channels, -1).contiguous()
    bias = proj.bias.contiguous() if proj.bias is not None else None

    def _fast_forward(self, hidden_states):
        return F.linear(hidden_states.to(weight.dtype), weight, bias)

    patch_embed.forward = types.MethodType(_fast_forward, patch_embed)
    log.info("Applied BidirLM-Omni patch-embed speed fix (Conv3d -> F.linear).")


def enforce_pixel_budget(model: models.ColBERT) -> None:
    """Force the image processor to honor MIN_PIXELS / MAX_PIXELS.

    BidirLMOmniProcessor keeps Qwen2VLImageProcessor's size = {shortest_edge,
    longest_edge} API but drops max_pixels when forwarded through the
    ColBERT/ST processor_kwargs chain, leaving longest_edge at the model-card
    default (~1M). We re-apply both bounds so processor_kwargs is authoritative.
    """
    processor = getattr(model[0], "processor", None)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        log.warning("Pixel-budget fix skipped (no image_processor).")
        return
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) or hasattr(size, "__setitem__"):
        size["shortest_edge"] = MIN_PIXELS
        size["longest_edge"] = MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = MIN_PIXELS
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    log.info(
        "Enforced pixel budget: shortest_edge=%d, longest_edge=%d.",
        MIN_PIXELS,
        MAX_PIXELS,
    )


def verify_chat_template(model: models.ColBERT) -> None:
    """Render a document (image+prompt) and a text-only query and log both.

    Renders through PyLate's resolved ``processing_kwargs["chat_template"]`` so
    the check reflects exactly what training will use — the forged slot for
    "colpali", or BidirLM-Omni's shipped template for "native".
    """
    from PIL import Image

    processor = model[0].processor
    kwargs = getattr(model[0], "processing_kwargs", {}).get("chat_template", {})
    dummy = Image.new("RGB", (28, 28), color="white")

    # Documents carry the prompt (as seer does); queries are bare text.
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
    query_render = processor.apply_chat_template(query_messages, tokenize=False, **kwargs)
    log.info(
        "Chat-template render check (variant=%r):\n  DOCUMENT:\n%s\n  QUERY:\n%s",
        CHAT_TEMPLATE_VARIANT,
        doc_render,
        query_render,
    )


def _short_dataset_name(repo: str) -> str:
    """``vidore/arxivqa_test_subsampled_beir`` -> ``arxivqa_test_subsampled``."""
    name = repo.split("/", 1)[-1]
    return name[: -len("_beir")] if name.endswith("_beir") else name


class _MacroEvaluator(SequentialEvaluator):
    """SequentialEvaluator that renames ``sequential_score`` to a macro key.

    Mirrors seer's ``ViDoRePyLateBenchmarkEvaluator`` so the two runs report the
    same wandb keys (e.g. ``vidore_v1_smoke_macro_MaxSim_ndcg@5``).
    """

    def __init__(self, sub_evaluators, *, name: str, primary_ndcg_k: int) -> None:
        super().__init__(sub_evaluators, main_score_function=mean)
        self.name = name
        self._macro_key = f"{name}_macro_MaxSim_ndcg@{primary_ndcg_k}"
        self.primary_metric = self._macro_key

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path, epoch, steps)
        results[self._macro_key] = results.pop("sequential_score")
        self.primary_metric = self._macro_key
        return results


def build_vidore_evaluator() -> SequentialEvaluator:
    """Build a ViDoRe v1 smoke evaluator using PyLate's MaxSim scoring."""
    log.info("Building ViDoRe evaluator...")
    sub_evaluators = []
    for repo in EVAL_DATASETS:
        log.info("Loading eval dataset %s...", repo)
        queries_ds = load_dataset(repo, "queries", split="test")
        corpus_ds = load_dataset(repo, "corpus", split="test")
        qrels_ds = load_dataset(repo, "qrels", split="test")

        queries = {str(r["query-id"]): r["query"] for r in queries_ds}
        corpus = {str(r["corpus-id"]): r["image"] for r in corpus_ds}
        relevant_docs: dict[str, set[str]] = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r["query-id"])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r["corpus-id"]))

        sub_name = f"{BENCHMARK_NAME}_{_short_dataset_name(repo)}"
        sub_evaluators.append(
            evaluation.PyLateInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=sub_name,
                batch_size=EVAL_BATCH_SIZE,
                corpus_chunk_size=EVAL_CORPUS_CHUNK_SIZE,
                ndcg_at_k=[NDCG_K],
            )
        )

    log.info("ViDoRe evaluator ready (%d sub-evaluators).", len(sub_evaluators))
    return _MacroEvaluator(sub_evaluators, name=BENCHMARK_NAME, primary_ndcg_k=NDCG_K)


def load_images_dataset() -> tuple[Dataset, dict[str, int]]:
    """Load the shared image dataset once, with lazy decoding."""
    log.info("Loading image dataset %s...", IMAGE_HUB_DATASET)
    images_ds = load_dataset(IMAGE_HUB_DATASET, split="train")
    images_ds = images_ds.cast_column("image", DatasetImage(decode=False))
    log.info("Building image index (%d images)...", len(images_ds))
    filenames = images_ds.with_format(None)["image_filename"]
    image_index = {name: i for i, name in enumerate(filenames)}
    log.info("Image index ready.")
    return images_ds, image_index


def load_split(
    split: str,
    images_ds: Dataset,
    image_index: dict[str, int],
) -> Dataset:
    """Load one KD split as a lazy-decoded dataset (query + positive + 1 negative)."""
    log.info("Loading split %s...", split)
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
        pos_filename = all_doc_filenames[doc_index[doc_ids[0]]]
        neg_filename = all_doc_filenames[doc_index[doc_ids[1]]]

        if pos_filename not in image_index or neg_filename not in image_index:
            continue

        rows["query"].append(query_text)
        rows["positive_image_idx"].append(image_index[pos_filename])
        rows["negative_image_idx"].append(image_index[neg_filename])

    ds = Dataset.from_dict(rows)

    decode_image = DatasetImage(decode=True)

    def transform(batch):
        # Wrap documents as {"image": pil, "text": DOCUMENT_PROMPT} like seer's
        # _format_document_value. BidirLM-Omni's shipped (native) template has no
        # DOCUMENT_PROMPT fallback, so for the "native" variant the prompt MUST
        # come from the data — otherwise raw-PIL docs render with no instruction.
        # The query stays bare text.
        out = {"query": batch["query"], "image": [], "negative_0": []}
        for pos_idx, neg_idx in zip(
            batch["positive_image_idx"], batch["negative_image_idx"]
        ):
            pos_pil = decode_image.decode_example(images_ds[pos_idx]["image"])
            neg_pil = decode_image.decode_example(images_ds[neg_idx]["image"])
            out["image"].append({"image": pos_pil, "text": DOCUMENT_PROMPT})
            out["negative_0"].append({"image": neg_pil, "text": DOCUMENT_PROMPT})
        return out

    ds.set_transform(transform)
    log.info("Split %s ready (%d rows).", split, len(ds))
    return ds


def main() -> None:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # ── 1. Model ─────────────────────────────────────────────────────────────
    # Chat template is fed at construction via processor_kwargs["chat_template"]:
    # PyLate's _configure_chat_template case (1) installs it under the
    # "sentence_transformers" slot and rewires apply_chat_template to use it.
    # For "native" we pass nothing → PyLate leaves BidirLM-Omni's shipped
    # template untouched (its no-system branch renders the bare text we want).
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "model_max_length": MAX_SEQ_LENGTH,
    }
    if CHAT_TEMPLATE is not None:  # "colpali" variant only
        processor_kwargs["chat_template"] = {"chat_template": CHAT_TEMPLATE}

    log.info("Loading model %s (chat template: %s)...", MODEL_NAME, CHAT_TEMPLATE_VARIANT)
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
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

    model[0].unpad_inputs = False

    # ── 1b. Model fixups ─────────────────────────────────────────────────────
    patch_bidirlm_omni_patch_embed(model)  # critical speed fix
    enforce_pixel_budget(model)
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
        init_lora_weights=True,
        use_rslora=True,
        use_dora=False,
    )
    model.add_adapter(lora_config)

    log.info("LoRA adapter added.")

    # ── 3. Data ──────────────────────────────────────────────────────────────
    images_ds, image_index = load_images_dataset()
    train_datasets = DatasetDict(
        {split: load_split(split, images_ds, image_index) for split in SPLITS}
    )

    log.info("All training splits loaded.")

    # ── 4. Loss ──────────────────────────────────────────────────────────────
    # gather_across_devices=True matches the seer config. NOTE: torch.distributed
    # is NOT yet initialized at this point under `accelerate launch` (the trainer
    # sets it up later), so checking dist.is_initialized() here returns False and
    # would silently disable cross-GPU negative gathering — shrinking the negative
    # pool 8x and producing a much weaker model. Hardcode True for multi-GPU runs.
    loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=MINI_BATCH_SIZE,
        gather_across_devices=True,
        temperature=TEMPERATURE,
    )

    # ── 5. Collator ──────────────────────────────────────────────────────────
    data_collator = utils.ColBERTCollator(preprocess_fn=model.preprocess)

    # ── 6. Training args ─────────────────────────────────────────────────────
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
        save_steps=10000,
        save_total_limit=2,
        logging_steps=1,
        seed=SEED,
        dataloader_num_workers=3,
        ddp_timeout=DDP_TIMEOUT,
        report_to=["wandb"],
    )

    # ── 7. Evaluator ────────────────────────────────────────────────────────
    evaluator = build_vidore_evaluator()
    # log.info("Running pre-training evaluation...")
    # evaluator(model)
    # log.info("Pre-training evaluation done.")

    # ── 8. Train ─────────────────────────────────────────────────────────────
    log.info("Initializing trainer...")
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

    # ── 9. Final eval + save ─────────────────────────────────────────────────
    log.info("Running post-training evaluation...")
    evaluator(model)
    log.info("Saving model to %s...", f"{OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    log.info("Done.")


if __name__ == "__main__":
    main()
