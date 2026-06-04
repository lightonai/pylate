"""Minimal ColPali-style (multi-vector) training for Qwen3.5-35B-A3B (MoE) via PyLate.

Counterpart to ``colpali_simple_qwen35.py`` for the sparse MoE backbone
(Qwen3.5-35B-A3B: 256 experts, top-8, ~3B active params).

READ BEFORE RUNNING — two MoE-specific realities:

1. LoRA targets. The dense paths (attention + GatedDeltaNet + shared_expert) are
   nn.Linear -> `target_modules`. The 256 routed experts are FUSED 3-D parameter
   tensors inside ``Qwen3_5MoeExperts`` (``gate_up_proj``: [256, 1024, 2048],
   ``down_proj``: [256, 2048, 512]) run through a grouped matmul — NOT nn.Linear.
   PEFT >= 0.19 can still adapt them via `target_parameters` (parameter-level
   LoRA), which we use by default (LORA_TARGET_EXPERTS=True). This is sound — the
   experts co-adapt while routing stays frozen (no router-instability risk).
   Cost: ~9.6M trainable (linears only) -> ~471M with experts. Set
   LORA_TARGET_EXPERTS=False for the lightweight, experts-frozen baseline.

2. MEMORY: ~35B params ≈ 70 GB bf16. Under plain DDP (accelerate launch) every
   GPU holds the full 70 GB, leaving little for activations + vision tokens on an
   80 GB card. Use gradient_checkpointing (on by default here) and
   mini_batch_size=1, and strongly prefer FSDP / DeepSpeed ZeRO-3 to SHARD the
   backbone rather than plain DDP. Also: the LM-tower fix's fallback reload would
   load a SECOND 35B copy — it's conditional (skips when the tower loads fine, as
   it does for the 4B) but would OOM hard if it ever triggers at this size.

Usage (plain DDP — may not fit; see above):
    accelerate launch --mixed_precision=bf16 scripts/train/colpali_simple_qwen35_moe.py
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
# "native"  — Qwen3.5's own chat template (no system prompt by default).
# "colpali" — ColPali-faithful (raw query, doc terminated with <|endoftext|>).
CHAT_TEMPLATE_VARIANT = "native"  # "native" | "colpali"

RUN_NAME = f"qwen3_5_35b_a3b_colpali_contrastive_simple_{CHAT_TEMPLATE_VARIANT}"
OUTPUT_DIR = f"models/{RUN_NAME}"
WANDB_PROJECT = "multimodal_pylate"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784  # 28 * 28
MAX_PIXELS = 360_000
MAX_SEQ_LENGTH = 4096

# ── Chat template ──────────────────────────────────────────────────────────--
DOCUMENT_PROMPT = "Describe the image."
#
# "native"  — DO NOT FORGE. Qwen3.5's shipped template; no system message ->
#   bare text (query: <|im_start|>user\n{q}<|im_end|>\n; document:
#   <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{doc text}<|im_end|>\n).
# "colpali" — ColPali-faithful override (raw query, doc ends <|endoftext|>).
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
CHAT_TEMPLATE = QWEN_VL_COLPALI_CHAT_TEMPLATE if CHAT_TEMPLATE_VARIANT == "colpali" else None

# ── LoRA config ──────────────────────────────────────────────────────────────
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# Qwen3.5-35B-A3B is HYBRID: 10 full-attention layers + 30 GatedDeltaNet
# (linear-attention) layers, every layer with a MoE block (256 routed experts +
# 1 shared expert). target_modules covers the nn.Linear token-mixing paths in
# EVERY layer, plus the always-active shared_expert:
#   - q/k/v/o_proj          -> the 10 full-attention layers
#   - in_proj_qkv/in_proj_z/out_proj -> the 30 GatedDeltaNet layers
#   - gate_proj/up_proj/down_proj    -> shared_expert (all 40 layers)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
# The 256 ROUTED experts are FUSED 3-D parameter tensors (Qwen3_5MoeExperts:
# gate_up_proj / down_proj) — not nn.Linear. PEFT >= 0.19 can still LoRA them via
# `target_parameters` (parameter-level adapters). This is sound: the experts
# co-adapt through their deltas while routing stays frozen (no router-instability
# / load-collapse risk). Cost scales with LORA_R and is dominated by the experts:
#   r=32 (default here): ~38M trainable (linears only) -> ~1.9B with experts (~5% of 35B)
#   r=8:                 ~9.6M                          -> ~471M
# FSDP-shardable either way. The per-expert MLP is small (moe_intermediate=512),
# so a high rank may over-parameterize the experts — consider a smaller rank (or
# rank_pattern) if you enable them. Set False for the lightweight frozen-expert baseline.
LORA_TARGET_EXPERTS = True
LORA_TARGET_PARAMETERS = (
    ["experts.gate_up_proj", "experts.down_proj"] if LORA_TARGET_EXPERTS else None
)
# No exclude needed: the vision tower uses attn.qkv / attn.proj /
# mlp.linear_fc1 / mlp.linear_fc2 (verified), which never collide with the
# q/k/v/o_proj (or gate/up/down_proj) target names — so it stays frozen
# without an explicit exclude. A "visual.blocks.*" exclude here matches nothing
# and just emits a PEFT warning. ("experts.down_proj" in target_parameters is
# specific — "shared_expert.down_proj" is singular and won't match.)
LORA_EXCLUDE_MODULES = None

# ── Loss config ──────────────────────────────────────────────────────────────
TEMPERATURE = 0.02

# ── Trainer config ───────────────────────────────────────────────────────────
BATCH_SIZE = 64  # global (split across GPUs). Reduce if OOM at 35B.
MINI_BATCH_SIZE = 1  # GradCache minibatch — keep at 1 for the 35B backbone.
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.005
MAX_STEPS = 3125
SEED = 42
EVAL_STEPS = 200
DDP_TIMEOUT = 14400  # 4h
GRADIENT_CHECKPOINTING = True  # required to fit activations at 35B

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

    AutoModel can route qwen3_5 / qwen3_5_moe to a bare model whose nested
    base_model_prefix="model" collision drops every `language_model.*` weight,
    leaving the LM tower at _init_weights defaults (input_layernorm.weight == 0).
    We probe and only reload when that actually happened.

    WARNING at 35B: the reload loads a SECOND ~70 GB copy and will OOM. It is
    conditional (skips when the tower loaded fine, as it does for the 4B), but if
    it ever triggers here you must fix the load differently (low-mem / in-place).
    Must run BEFORE LoRA is attached.
    """
    transformer = model[0]
    probe = transformer.model.language_model.layers[0].input_layernorm.weight
    if float(probe.abs().sum().item()) != 0.0:
        log.info("Qwen3.5-MoE LM tower loaded correctly (layernorm sum != 0) — no fix needed.")
        return

    log.warning("Qwen3.5-MoE LM tower loaded BROKEN — reloading via "
                "AutoModelForImageTextToText (WARNING: loads a 2nd ~70GB copy).")
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
            "Qwen3.5-MoE language_model weights still uninitialised after reload — "
            "the multimodal loading path has regressed."
        )
    log.info("Applied Qwen3.5-MoE LM-tower fix (reloaded via AutoModelForImageTextToText).")


def verify_chat_template(model: models.ColBERT) -> None:
    """Render a document (image+prompt) and a text-only query and log both."""
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

        sub_name = f"{BENCHMARK_NAME}:{_short_dataset_name(repo)}"
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
        # Wrap documents with DOCUMENT_PROMPT (the native shipped template has no
        # fallback, so the prompt must come from the data). Query stays bare text.
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
    # No pixel-budget fixup needed — Qwen3VLProcessor honors min/max_pixels from
    # processor_kwargs (verified: longest_edge=360000 at load).
    fix_qwen35_backbone_weights(model)  # only reloads if the LM tower loaded broken
    model[0].unpad_inputs = False
    verify_chat_template(model)
    log.info("Model loaded.")

    # ── 2. LoRA ───────────────────────────────────────────────────────────────
    # target_modules -> nn.Linear paths (all 40 layers + shared_expert).
    # target_parameters -> the fused routed-expert tensors (PEFT >= 0.19), when
    # LORA_TARGET_EXPERTS is on.
    log.info(
        "Adding LoRA adapter (modules=%s, target_parameters=%s)...",
        LORA_TARGET_MODULES,
        LORA_TARGET_PARAMETERS,
    )
    lora_kwargs = dict(
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
    if LORA_TARGET_PARAMETERS is not None:
        lora_kwargs["target_parameters"] = LORA_TARGET_PARAMETERS
    lora_config = LoraConfig(**lora_kwargs)
    model.add_adapter(lora_config)
    log.info("LoRA adapter added.")

    # ── 3. Data ──────────────────────────────────────────────────────────────
    images_ds, image_index = load_images_dataset()
    train_datasets = DatasetDict(
        {split: load_split(split, images_ds, image_index) for split in SPLITS}
    )
    log.info("All training splits loaded.")

    # ── 4. Loss ──────────────────────────────────────────────────────────────
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
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
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
