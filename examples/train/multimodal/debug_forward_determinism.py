"""Check if the same model produces identical outputs across repeated calls.
Tests whether flash_attention_2 introduces non-determinism.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_forward_determinism.py
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


def main():
    log.info("Loading model from %s", CHECKPOINT_PATH)
    model = models.ColBERT(
        model_name_or_path=CHECKPOINT_PATH,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH},
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model[0].unpad_inputs = False
    model.eval()

    dummy_image = Image.new("RGB", (56, 56), color="red")
    doc_input = {"image": dummy_image, "text": "Describe the image."}
    query_input = "What is shown in this image?"

    N = 5
    log.info("Running %d forward passes with identical input...", N)

    doc_embeddings = []
    query_embeddings = []
    for i in range(N):
        with torch.no_grad():
            d = model.encode([doc_input], is_query=False, batch_size=1)
            q = model.encode([query_input], is_query=True, batch_size=1)
        doc_embeddings.append(torch.tensor(d[0]).float())
        query_embeddings.append(torch.tensor(q[0]).float())

    log.info("=" * 70)
    log.info("DOC EMBEDDING STABILITY (same model, same input, %d runs)", N)
    log.info("=" * 70)
    ref = doc_embeddings[0]
    for i in range(1, N):
        delta = (doc_embeddings[i] - ref).abs()
        cos = torch.nn.functional.cosine_similarity(
            doc_embeddings[i].flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        log.info("  run 0 vs run %d: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
                 i, delta.max().item(), delta.mean().item(), cos)

    log.info("=" * 70)
    log.info("QUERY EMBEDDING STABILITY (same model, same input, %d runs)", N)
    log.info("=" * 70)
    ref = query_embeddings[0]
    for i in range(1, N):
        delta = (query_embeddings[i] - ref).abs()
        cos = torch.nn.functional.cosine_similarity(
            query_embeddings[i].flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        log.info("  run 0 vs run %d: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
                 i, delta.max().item(), delta.mean().item(), cos)

    # Now test with sdpa instead of flash_attention_2
    log.info("=" * 70)
    log.info("TESTING WITH sdpa ATTENTION (deterministic)")
    log.info("=" * 70)
    del model
    torch.cuda.empty_cache()

    model_sdpa = models.ColBERT(
        model_name_or_path=CHECKPOINT_PATH,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": TORCH_DTYPE},
        processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH},
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model_sdpa[0].unpad_inputs = False
    model_sdpa.eval()

    doc_sdpa = []
    query_sdpa = []
    for i in range(N):
        with torch.no_grad():
            d = model_sdpa.encode([doc_input], is_query=False, batch_size=1)
            q = model_sdpa.encode([query_input], is_query=True, batch_size=1)
        doc_sdpa.append(torch.tensor(d[0]).float())
        query_sdpa.append(torch.tensor(q[0]).float())

    log.info("DOC EMBEDDING STABILITY (sdpa, %d runs):", N)
    ref = doc_sdpa[0]
    for i in range(1, N):
        delta = (doc_sdpa[i] - ref).abs()
        cos = torch.nn.functional.cosine_similarity(
            doc_sdpa[i].flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
        log.info("  run 0 vs run %d: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
                 i, delta.max().item(), delta.mean().item(), cos)

    log.info("QUERY EMBEDDING STABILITY (sdpa, %d runs):", N)
    ref = query_sdpa[0]
    for i in range(1, N):
        delta = (query_sdpa[i] - ref).abs()
        cos = torch.nn.functional.cosine_similarity(
            query_sdpa[i].flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
        log.info("  run 0 vs run %d: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
                 i, delta.max().item(), delta.mean().item(), cos)

    # Cross-compare: flash vs sdpa (should differ due to different attn implementations)
    log.info("=" * 70)
    log.info("CROSS: flash_attention_2 vs sdpa (same weights)")
    log.info("=" * 70)
    # Reload flash model
    del model_sdpa
    torch.cuda.empty_cache()

    model_flash = models.ColBERT(
        model_name_or_path=CHECKPOINT_PATH,
        model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": TORCH_DTYPE},
        processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "model_max_length": MAX_SEQ_LENGTH},
        trust_remote_code=True,
        embedding_size=EMBEDDING_SIZE,
        query_prefix="", document_prefix="",
        query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH,
        do_query_expansion=True, attend_to_expansion_tokens=True,
    )
    model_flash[0].unpad_inputs = False
    model_flash.eval()

    with torch.no_grad():
        doc_flash = torch.tensor(model_flash.encode([doc_input], is_query=False, batch_size=1)[0]).float()
        query_flash = torch.tensor(model_flash.encode([query_input], is_query=True, batch_size=1)[0]).float()

    delta_d = (doc_flash - doc_sdpa[0]).abs()
    delta_q = (query_flash - query_sdpa[0]).abs()
    log.info("  doc: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta_d.max().item(), delta_d.mean().item(),
             torch.nn.functional.cosine_similarity(
                 doc_flash.flatten().unsqueeze(0), doc_sdpa[0].flatten().unsqueeze(0)).item())
    log.info("  query: max_diff=%.2e, mean_diff=%.2e, cosine=%.8f",
             delta_q.max().item(), delta_q.mean().item(),
             torch.nn.functional.cosine_similarity(
                 query_flash.flatten().unsqueeze(0), query_sdpa[0].flatten().unsqueeze(0)).item())


if __name__ == "__main__":
    main()
