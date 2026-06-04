"""Minimal ColPali-style (multi-vector) training for Qwen3.5-4B via PyLate.

Self-contained counterpart to ``colpali_simple_bidirlm.py``, configured for the
Qwen3.5-4B backbone. Trains a multi-vector ColBERT (MaxSim) with the same KD
colpali data and CachedContrastive loss as the other colpali_simple_* scripts.
(Note: seer's own Qwen3.5 configs train a DENSE embedder; this is the
multi-vector experiment, mirroring the colpali scripts.)

Key model-specific piece — the FIX:
  * Qwen3.5's model_type is "qwen3_5" with architecture
    Qwen3_5ForConditionalGeneration (NOT a ColPali arch), so PyLate/ST load the
    backbone via AutoModel.from_pretrained -> bare Qwen3_5Model. Both
    Qwen3_5Model and its inner Qwen3_5TextModel declare base_model_prefix="model",
    and the nested collision makes prefix-stripping miss every
    `language_model.*` weight: THE LM TOWER LOADS AT _init_weights DEFAULTS
    (silently broken). We fix it by reloading through
    AutoModelForImageTextToText (-> Qwen3_5ForConditionalGeneration, whose
    `.model` is a correctly-loaded Qwen3_5Model) and swapping it into the ST
    Transformer module. Ported from seer.models._fix_qwen35_backbone_weights.

Usage:
    accelerate launch --mixed_precision=bf16 scripts/train/colpali_simple_qwen35.py
"""

from __future__ import annotations

import logging
import os
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import torch
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
from transformers import AutoModelForImageTextToText

# ── Run config ───────────────────────────────────────────────────────────────
# "native"  — Qwen3.5's own chat template (no system prompt by default). This is
#             what seer training uses (model_type "qwen3_5" isn't in PyLate's
#             COLPALI_CHAT_TEMPLATES, so the shipped template stands).
# "colpali" — ColPali-faithful (raw query, doc terminated with <|endoftext|>).
CHAT_TEMPLATE_VARIANT = "native"  # "native" | "colpali"

RUN_NAME = f"qwen3_5_4b_colpali_contrastive_simple_{CHAT_TEMPLATE_VARIANT}"
OUTPUT_DIR = f"models/{RUN_NAME}"
WANDB_PROJECT = "multimodal_pylate"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784  # 28 * 28
MAX_PIXELS = 360_000  # per seer's qwen3_5 configs
MAX_SEQ_LENGTH = 4096

# ── Chat template ──────────────────────────────────────────────────────────--
# Document instruction text. Seer feeds documents this prompt; with the native
# (shipped) template the prompt MUST come from the data (the shipped template has
# no fallback), so the transform wraps documents with it.
DOCUMENT_PROMPT = "Describe the image."
#
# "native"  — DO NOT FORGE. Use Qwen3.5's shipped chat template. With no system
#   message its branch renders bare text:
#       query:    <|im_start|>user\n{query}<|im_end|>\n
#       document: <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{doc text}<|im_end|>\n
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
LORA_TARGET_MODULES = [
    "down_proj",
    "gate_proj",
    "up_proj",
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
]
LORA_EXCLUDE_MODULES = "visual.blocks.*"

# ── Loss config ──────────────────────────────────────────────────────────────
TEMPERATURE = 0.02

# ── Trainer config ───────────────────────────────────────────────────────────
BATCH_SIZE = 64
MINI_BATCH_SIZE = 4  # 4B backbone — keep GradCache minibatch small
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.005
MAX_STEPS = 3125
SEED = 42
EVAL_STEPS = 200
DDP_TIMEOUT = 14400  # 4h — rank-0 ViDoRe eval on the 4B backbone.

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


# ── Model fixup (ported from seer.models._fix_qwen35_backbone_weights) ───────--
def fix_qwen35_backbone_weights(model: models.ColBERT) -> None:
    """Reload the LM tower via AutoModelForImageTextToText IF it loaded broken.

    Background: AutoModel.from_pretrained can route "qwen3_5" to a bare
    Qwen3_5Model whose nested base_model_prefix="model" collision makes
    prefix-stripping miss every `language_model.*` weight, leaving the LM tower
    at _init_weights defaults (input_layernorm.weight == 0). When that happens we
    reload through AutoModelForImageTextToText (-> Qwen3_5ForConditionalGeneration,
    whose `.model` is a correctly-loaded Qwen3_5Model) and swap it in.

    This is CONDITIONAL: in the current PyLate path + transformers version the
    tower loads fine, so we probe first and only reload when actually broken —
    avoiding a wasteful second 4B load. Must run BEFORE LoRA is attached.
    """
    transformer = model[0]
    probe = transformer.model.language_model.layers[0].input_layernorm.weight
    if float(probe.abs().sum().item()) != 0.0:
        log.info("Qwen3.5 LM tower loaded correctly (layernorm sum != 0) — no fix needed.")
        return

    log.warning("Qwen3.5 LM tower loaded BROKEN (layernorm sum == 0) — reloading via "
                "AutoModelForImageTextToText.")
    wrapper = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
    )
    target_device = next(transformer.model.parameters()).device
    wrapper = wrapper.to(target_device)
    transformer.model = wrapper.model
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
    """Render a document (image+prompt) and a text-only query and log both.

    Renders through PyLate's resolved ``processing_kwargs["chat_template"]`` so
    the check reflects exactly what training will use.
    """
    from PIL import Image

    processor = model[0].processor
    kwargs = getattr(model[0], "processing_kwargs", {}).get("chat_template", {})
    dummy = Image.new("RGB", (28, 28), color="white")

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
    """SequentialEvaluator that renames ``sequential_score`` to a macro key."""

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
        # _format_document_value. The native (shipped) template has no
        # DOCUMENT_PROMPT fallback, so the prompt must come from the data. The
        # query stays bare text.
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
    # Chat template fed at construction via processor_kwargs["chat_template"]
    # (PyLate case (1)). For "native" we pass nothing -> shipped template stands.
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

    # ── 1b. Model fixups (BEFORE LoRA) ───────────────────────────────────────
    # NOTE: no pixel-budget fixup needed — Qwen3VLProcessor honors min/max_pixels
    # straight from processor_kwargs (verified: longest_edge=360000 at load).
    # Unlike BidirLMOmniProcessor, nothing is dropped through the kwargs chain.
    fix_qwen35_backbone_weights(model)  # only reloads if the LM tower loaded broken
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
    # would silently disable cross-GPU negative gathering. Hardcode True.
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
