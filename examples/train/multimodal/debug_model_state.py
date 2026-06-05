"""Deep comparison of model state beyond just parameters.
Checks buffers, module structure, LoRA wiring, and autocast effects.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_model_state.py
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from PIL import Image
from pylate import models

CHECKPOINT_PATH = "models/qwen3_5_4b_colpali_contrastive_simple_native/checkpoint-3125"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784
MAX_PIXELS = 360_000
MAX_SEQ_LENGTH = 4096


def load_model(path):
    m = models.ColBERT(
        model_name_or_path=path,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH},
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    m[0].unpad_inputs = False
    m.eval()
    return m


def compare_buffers(model_a, model_b, label_a="A", label_b="B"):
    log.info("=" * 70)
    log.info("BUFFER COMPARISON: %s vs %s", label_a, label_b)
    log.info("=" * 70)
    bufs_a = dict(model_a.named_buffers())
    bufs_b = dict(model_b.named_buffers())
    only_a = set(bufs_a) - set(bufs_b)
    only_b = set(bufs_b) - set(bufs_a)
    common = set(bufs_a) & set(bufs_b)
    log.info("  %s buffers: %d, %s buffers: %d, common: %d", label_a, len(bufs_a), label_b, len(bufs_b), len(common))
    if only_a:
        log.warning("  Only in %s: %s", label_a, sorted(only_a)[:10])
    if only_b:
        log.warning("  Only in %s: %s", label_b, sorted(only_b)[:10])
    diffs = 0
    for k in sorted(common):
        a, b = bufs_a[k].float(), bufs_b[k].float()
        if a.shape != b.shape:
            log.error("  SHAPE: %s %s vs %s", k, a.shape, b.shape)
            diffs += 1
        elif (a - b).abs().max().item() > 0:
            log.warning("  DIFF: %s max=%.2e", k, (a - b).abs().max().item())
            diffs += 1
    if diffs == 0:
        log.info("  All common buffers match exactly.")


def compare_module_structure(model_a, model_b, label_a="A", label_b="B"):
    log.info("=" * 70)
    log.info("MODULE STRUCTURE: %s vs %s", label_a, label_b)
    log.info("=" * 70)
    mods_a = {n: type(m).__name__ for n, m in model_a.named_modules()}
    mods_b = {n: type(m).__name__ for n, m in model_b.named_modules()}
    only_a = set(mods_a) - set(mods_b)
    only_b = set(mods_b) - set(mods_a)
    log.info("  %s modules: %d, %s modules: %d", label_a, len(mods_a), label_b, len(mods_b))
    if only_a:
        log.warning("  Only in %s (%d): %s", label_a, len(only_a), sorted(only_a)[:10])
    if only_b:
        log.warning("  Only in %s (%d): %s", label_b, len(only_b), sorted(only_b)[:10])
    type_mismatches = [(n, mods_a[n], mods_b[n]) for n in (set(mods_a) & set(mods_b)) if mods_a[n] != mods_b[n]]
    if type_mismatches:
        log.warning("  Type mismatches:")
        for n, ta, tb in type_mismatches[:10]:
            log.warning("    %s: %s vs %s", n, ta, tb)


def check_lora_details(model, label):
    log.info("=" * 70)
    log.info("LORA DETAILS: %s", label)
    log.info("=" * 70)
    auto = model[0].auto_model
    log.info("  _hf_peft_config_loaded: %s", getattr(auto, "_hf_peft_config_loaded", "N/A"))
    log.info("  active_adapters: %s", getattr(auto, "active_adapters", "N/A"))
    log.info("  peft_config: %s", getattr(auto, "peft_config", "N/A"))

    from peft.tuners.tuners_utils import BaseTunerLayer
    lora_count = 0
    for name, module in auto.named_modules():
        if isinstance(module, BaseTunerLayer):
            lora_count += 1
            if lora_count <= 2:
                log.info("  LoRA layer %s:", name)
                log.info("    type: %s", type(module).__name__)
                log.info("    active_adapter: %s", getattr(module, "active_adapter", "N/A"))
                log.info("    scaling: %s", getattr(module, "scaling", "N/A"))
                log.info("    merged: %s", getattr(module, "merged", "N/A"))
                log.info("    in_features: %s, out_features: %s",
                         getattr(module, "in_features", "N/A"),
                         getattr(module, "out_features", "N/A"))
                if hasattr(module, "lora_A"):
                    for adapter_name, lora_a in module.lora_A.items():
                        log.info("    lora_A[%s]: %s, dtype=%s", adapter_name, lora_a.weight.shape, lora_a.weight.dtype)
                if hasattr(module, "lora_B"):
                    for adapter_name, lora_b in module.lora_B.items():
                        log.info("    lora_B[%s]: %s, dtype=%s", adapter_name, lora_b.weight.shape, lora_b.weight.dtype)
                # Check the base layer
                if hasattr(module, "base_layer"):
                    bl = module.base_layer
                    log.info("    base_layer: %s, dtype=%s", type(bl).__name__, bl.weight.dtype)
    log.info("  Total LoRA layers: %d", lora_count)


def compare_embeddings(model_a, model_b, label_a="A", label_b="B"):
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    log.info("=" * 70)
    log.info("EMBEDDING COMPARISON: %s vs %s", label_a, label_b)
    log.info("=" * 70)

    with torch.no_grad():
        d_a = torch.tensor(model_a.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        d_b = torch.tensor(model_b.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        q_a = torch.tensor(model_a.encode([query_input], is_query=True, batch_size=1)[0]).float()
        q_b = torch.tensor(model_b.encode([query_input], is_query=True, batch_size=1)[0]).float()

    for name, a, b in [("doc", d_a, d_b), ("query", q_a, q_b)]:
        delta = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
        log.info("  %s: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f", name, delta.max().item(), delta.mean().item(), cos)


def check_autocast_effect(model, label):
    log.info("=" * 70)
    log.info("AUTOCAST EFFECT: %s", label)
    log.info("=" * 70)
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}

    with torch.no_grad():
        emb_no_cast = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        emb_with_cast = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta = (emb_no_cast - emb_with_cast).abs()
    cos = torch.nn.functional.cosine_similarity(emb_no_cast.flatten().unsqueeze(0), emb_with_cast.flatten().unsqueeze(0)).item()
    log.info("  no_autocast vs bf16_autocast: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta.max().item(), delta.mean().item(), cos)


def main():
    log.info("Test 1: Load SAME checkpoint TWICE, compare everything")
    log.info("(If these differ, the loading path itself is non-deterministic)")

    model_a = load_model(CHECKPOINT_PATH)
    model_b = load_model(CHECKPOINT_PATH)

    compare_buffers(model_a, model_b, "load1", "load2")
    compare_module_structure(model_a, model_b, "load1", "load2")
    check_lora_details(model_a, "load1")
    check_lora_details(model_b, "load2")
    compare_embeddings(model_a, model_b, "load1", "load2")

    log.info("")
    log.info("Test 2: Autocast effect on loaded model")
    check_autocast_effect(model_a, "load1")

    # Test 3: Check if training mode changes anything
    log.info("")
    log.info("=" * 70)
    log.info("Test 3: EVAL vs TRAIN mode embedding comparison")
    log.info("=" * 70)
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}

    model_a.eval()
    with torch.no_grad():
        emb_eval = torch.tensor(model_a.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    model_a.train()
    with torch.no_grad():
        emb_train = torch.tensor(model_a.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta = (emb_eval - emb_train).abs()
    cos = torch.nn.functional.cosine_similarity(emb_eval.flatten().unsqueeze(0), emb_train.flatten().unsqueeze(0)).item()
    log.info("  eval vs train mode: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta.max().item(), delta.mean().item(), cos)

    model_a.eval()

    # Test 4: Check if bf16 autocast during training eval creates the gap
    log.info("")
    log.info("=" * 70)
    log.info("Test 4: Simulating training-time eval (with bf16 autocast like trainer)")
    log.info("=" * 70)
    # SentenceTransformerTrainer uses bf16 autocast during eval
    # Let's check if that changes results

    model_a.eval()
    with torch.no_grad():
        emb_plain = torch.tensor(model_a.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    # Simulate what trainer does: sets compute_loss with autocast
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        emb_autocast = torch.tensor(model_a.encode([doc_input], is_query=False, batch_size=1)[0]).float()

    delta = (emb_plain - emb_autocast).abs()
    cos = torch.nn.functional.cosine_similarity(emb_plain.flatten().unsqueeze(0), emb_autocast.flatten().unsqueeze(0)).item()
    log.info("  plain vs bf16_autocast: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta.max().item(), delta.mean().item(), cos)


if __name__ == "__main__":
    main()
