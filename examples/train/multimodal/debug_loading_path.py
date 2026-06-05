"""Test if the difference exists before training — just from the loading path.

Path A: Load base model + add_adapter() (how training sets up)
Path B: Save adapter + reload (how inference loads)

If these differ WITHOUT any training, the issue is in the loading path.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_loading_path.py
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
from peft import LoraConfig, TaskType
from pylate import models

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "models/_debug_loading_path"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
EMBEDDING_SIZE = 128
QUERY_LENGTH = 48
DOCUMENT_LENGTH = 1024
MIN_PIXELS = 784
MAX_PIXELS = 360_000
MAX_SEQ_LENGTH = 4096

LORA_R = 32
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["down_proj", "gate_proj", "up_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
LORA_EXCLUDE_MODULES = "visual.blocks.*"


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
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH}
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    # PATH A: Load base + add_adapter (training setup)
    log.info("PATH A: Loading base model + add_adapter()...")
    model_a = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model_a[0].unpad_inputs = False

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES, init_lora_weights=True,
        use_rslora=True, use_dora=False,
    )
    model_a.add_adapter(lora_config)

    emb_a = get_emb(model_a, doc_input, query_input)
    log.info("Path A embeddings captured")

    # Save Path A's model
    save_path = f"{OUTPUT_DIR}/adapter"
    model_a.save_pretrained(save_path)
    log.info("Saved Path A to %s", save_path)

    # PATH B: Load from saved adapter (inference setup)
    log.info("PATH B: Loading from saved adapter...")
    model_b = models.ColBERT(
        model_name_or_path=save_path,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model_b[0].unpad_inputs = False

    emb_b = get_emb(model_b, doc_input, query_input)
    log.info("Path B embeddings captured")

    # Compare
    log.info("=" * 70)
    log.info("COMPARISON: Path A (base+add_adapter) vs Path B (load from saved)")
    log.info("NO TRAINING WAS DONE — just save + reload")
    compare("A vs B", emb_a, emb_b)

    # Check params
    log.info("")
    log.info("PARAM CHECK:")
    sd_a = {k: v.cpu() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.cpu() for k, v in model_b.state_dict().items()}
    common = set(sd_a) & set(sd_b)
    diffs = []
    for k in sorted(common):
        a, b = sd_a[k].float(), sd_b[k].float()
        if a.shape == b.shape:
            md = (a - b).abs().max().item()
            if md > 0:
                diffs.append((k, md))
    log.info("  Common keys: %d, Diffs: %d", len(common), len(diffs))
    for k, md in diffs[:10]:
        log.info("    %s: max_diff=%.2e", k, md)

    # PATH C: Load from saved adapter, then reload AGAIN
    log.info("")
    log.info("PATH C: Loading Path B model's save (second save/reload cycle)...")
    save_path_2 = f"{OUTPUT_DIR}/adapter_2"
    model_b.save_pretrained(save_path_2)
    model_c = models.ColBERT(
        model_name_or_path=save_path_2,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs=processor_kwargs,
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model_c[0].unpad_inputs = False

    emb_c = get_emb(model_c, doc_input, query_input)
    log.info("=" * 70)
    log.info("B vs C (load vs load-save-load — should be identical):")
    compare("B vs C", emb_b, emb_c)


if __name__ == "__main__":
    main()
