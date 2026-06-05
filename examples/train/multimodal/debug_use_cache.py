"""Test if use_cache config difference causes embedding mismatch.

After training, trainer sets model.config.use_cache=False and doesn't
restore it. The reloaded model has use_cache=None (default). This test
checks whether aligning use_cache eliminates the embedding difference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_use_cache.py
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


def encode_and_compare(model, label, doc_input, query_input, ref_doc=None, ref_query=None):
    with torch.no_grad():
        d = torch.tensor(model.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        q = torch.tensor(model.encode([query_input], is_query=True, batch_size=1)[0]).float()
    log.info("  %s: doc norm=%.4f, query norm=%.4f", label, d.norm().item(), q.norm().item())
    if ref_doc is not None:
        dd = (d - ref_doc).abs()
        dq = (q - ref_query).abs()
        log.info("    vs ref: doc max_diff=%.2e mean=%.2e cos=%.8f",
                 dd.max().item(), dd.mean().item(),
                 torch.nn.functional.cosine_similarity(d.flatten().unsqueeze(0), ref_doc.flatten().unsqueeze(0)).item())
        log.info("    vs ref: query max_diff=%.2e mean=%.2e cos=%.8f",
                 dq.max().item(), dq.mean().item(),
                 torch.nn.functional.cosine_similarity(q.flatten().unsqueeze(0), ref_query.flatten().unsqueeze(0)).item())
    return d, q


def main():
    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    log.info("Loading model...")
    model = load_model(CHECKPOINT_PATH)

    cfg = model[0].auto_model.config
    use_cache = getattr(cfg, "use_cache", "NOT_SET")
    architectures = getattr(cfg, "architectures", "NOT_SET")
    log.info("Initial config: use_cache=%s, architectures=%s", use_cache, architectures)

    # Check sub-configs
    for sub_name in ["text_config", "vision_config"]:
        sub_cfg = getattr(cfg, sub_name, None)
        if sub_cfg:
            log.info("  %s.use_cache=%s", sub_name, getattr(sub_cfg, "use_cache", "NOT_SET"))

    # List all config attributes that differ from defaults
    log.info("  All config keys with 'cache': %s",
             [(k, v) for k, v in vars(cfg).items() if "cache" in k.lower()])

    # Check text_config use_cache to toggle
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        log.info("No text_config, checking top-level config")
        text_cfg = cfg

    uc_orig = getattr(text_cfg, "use_cache", "NOT_SET")
    log.info("Will toggle text_config.use_cache (currently=%s)", uc_orig)

    # Baseline: as-is
    log.info("")
    log.info("=" * 70)
    log.info("TEST 1: use_cache=%s (as loaded)", uc_orig)
    log.info("=" * 70)
    ref_doc, ref_query = encode_and_compare(model, f"use_cache={uc_orig}", doc_input, query_input)

    # Set use_cache=False
    log.info("")
    log.info("=" * 70)
    log.info("TEST 2: use_cache=False")
    log.info("=" * 70)
    text_cfg.use_cache = False
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    d_false, q_false = encode_and_compare(model, "use_cache=False", doc_input, query_input, ref_doc, ref_query)

    # Set use_cache=True
    log.info("")
    log.info("=" * 70)
    log.info("TEST 3: use_cache=True")
    log.info("=" * 70)
    text_cfg.use_cache = True
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = True
    d_true, q_true = encode_and_compare(model, "use_cache=True", doc_input, query_input, ref_doc, ref_query)

    # Restore
    log.info("")
    log.info("=" * 70)
    log.info("TEST 4: restored to original")
    log.info("=" * 70)
    text_cfg.use_cache = uc_orig if uc_orig != "NOT_SET" else True
    d_restore, q_restore = encode_and_compare(model, "restored", doc_input, query_input, ref_doc, ref_query)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("  orig vs False: doc max_diff=%.2e, query max_diff=%.2e",
             (ref_doc - d_false).abs().max().item(), (ref_query - q_false).abs().max().item())
    log.info("  orig vs True:  doc max_diff=%.2e, query max_diff=%.2e",
             (ref_doc - d_true).abs().max().item(), (ref_query - q_true).abs().max().item())
    log.info("  True vs False: doc max_diff=%.2e, query max_diff=%.2e",
             (d_true - d_false).abs().max().item(), (q_true - q_false).abs().max().item())


if __name__ == "__main__":
    main()
