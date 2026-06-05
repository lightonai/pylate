"""Isolate exactly where bf16 precision degrades embeddings.

Pipeline: Backbone(bf16) → Dense(bf16) → L2Norm(bf16)

Tests:
  1. Full bf16 (baseline)
  2. Cast to fp32 BEFORE Dense (backbone bf16, Dense+norm fp32)
  3. Cast to fp32 AFTER Dense, before L2 normalize
  4. FP32 backbone + bf16 Dense + bf16 norm
  5. Decomposed stage-by-stage error measurement

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_precision_isolation.py [model_path]
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.WARNING)

import torch
import torch.nn.functional as F
from PIL import Image
from pylate import models

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3_5_4b_colpali_contrastive_simple_native/final"
DOCUMENT_PROMPT = "Describe the image."


def compare(label, test_embs, ref_embs, kind="doc"):
    if not isinstance(test_embs, list):
        test_embs, ref_embs = [test_embs], [ref_embs]
    for i, (t, r) in enumerate(zip(test_embs, ref_embs)):
        t_f = torch.tensor(t).float().flatten()
        r_f = torch.tensor(r).float().flatten()
        delta = (t_f - r_f).abs()
        cos = F.cosine_similarity(t_f.unsqueeze(0), r_f.unsqueeze(0)).item()
        tag = f"{kind}[{i}]" if len(test_embs) > 1 else kind
        if delta.max().item() == 0:
            print(f"  {label:50s} {tag:10s} IDENTICAL")
        else:
            print(f"  {label:50s} {tag:10s} max={delta.max().item():.4e} mean={delta.mean().item():.4e} cos={cos:.8f}")


def encode_with_hook(model, doc_inputs, queries, dense_hook=None):
    """Encode with an optional hook around the Dense layer."""
    if dense_hook:
        orig = model[1].forward
        model[1].forward = lambda features: dense_hook(features, orig)

    with torch.no_grad():
        doc_emb = model.encode(doc_inputs, is_query=False, batch_size=1)
        query_embs = model.encode(queries, is_query=True, batch_size=4)

    if dense_hook:
        model[1].forward = orig

    return doc_emb, query_embs


def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dummy = Image.new("RGB", (224, 224), color="blue")
    doc_inputs = [{"image": dummy, "text": DOCUMENT_PROMPT}]
    queries = ["What is shown in this image?", "Describe the content of this page.", "What information is presented here?"]

    print(f"Model: {MODEL_DIR}")
    print("=" * 80)

    # === FP32 reference ===
    print("Loading fp32 reference...")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, model_kwargs={"torch_dtype": torch.float32}, device="cuda")
    model.eval()
    ref_doc, ref_q = encode_with_hook(model, doc_inputs, queries)
    del model; torch.cuda.empty_cache()

    # === TEST 1: Full bf16 ===
    print("\n--- TEST 1: Full BF16 (baseline) ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, device="cuda")
    model.eval()
    d, q = encode_with_hook(model, doc_inputs, queries)
    compare("full_bf16", d, ref_doc, "doc")
    compare("full_bf16", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 2: Cast to fp32 BEFORE Dense ===
    print("\n--- TEST 2: BF16 backbone → cast fp32 → FP32 Dense → FP32 norm ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, device="cuda")
    model.eval()
    model[1].float()

    def hook_cast_before(features, orig_fwd):
        features["token_embeddings"] = features["token_embeddings"].float()
        return orig_fwd(features)

    d, q = encode_with_hook(model, doc_inputs, queries, hook_cast_before)
    compare("bf16_backbone→fp32_dense→fp32_norm", d, ref_doc, "doc")
    compare("bf16_backbone→fp32_dense→fp32_norm", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 3: Cast to fp32 AFTER Dense (before normalize) ===
    print("\n--- TEST 3: BF16 backbone → BF16 Dense → cast fp32 → FP32 norm ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, device="cuda")
    model.eval()

    def hook_cast_after(features, orig_fwd):
        features = orig_fwd(features)
        features["token_embeddings"] = features["token_embeddings"].float()
        return features

    d, q = encode_with_hook(model, doc_inputs, queries, hook_cast_after)
    compare("bf16_backbone→bf16_dense→fp32_norm", d, ref_doc, "doc")
    compare("bf16_backbone→bf16_dense→fp32_norm", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 4: FP32 backbone + bf16 Dense + bf16 norm ===
    print("\n--- TEST 4: FP32 backbone → bf16 Dense → bf16 norm ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, model_kwargs={"torch_dtype": torch.float32}, device="cuda")
    model.eval()
    model[1].bfloat16()

    def hook_cast_to_bf16(features, orig_fwd):
        features["token_embeddings"] = features["token_embeddings"].bfloat16()
        return orig_fwd(features)

    d, q = encode_with_hook(model, doc_inputs, queries, hook_cast_to_bf16)
    compare("fp32_backbone→bf16_dense→bf16_norm", d, ref_doc, "doc")
    compare("fp32_backbone→bf16_dense→bf16_norm", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 5: FP32 backbone + bf16 Dense + FP32 norm ===
    print("\n--- TEST 5: FP32 backbone → bf16 Dense → FP32 norm (isolate Dense) ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, model_kwargs={"torch_dtype": torch.float32}, device="cuda")
    model.eval()
    model[1].bfloat16()

    def hook_bf16_dense_fp32_norm(features, orig_fwd):
        features["token_embeddings"] = features["token_embeddings"].bfloat16()
        features = orig_fwd(features)
        features["token_embeddings"] = features["token_embeddings"].float()
        return features

    d, q = encode_with_hook(model, doc_inputs, queries, hook_bf16_dense_fp32_norm)
    compare("fp32_backbone→bf16_dense→fp32_norm", d, ref_doc, "doc")
    compare("fp32_backbone→bf16_dense→fp32_norm", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 6: FP32 backbone + FP32 Dense + bf16 norm (isolate normalize) ===
    print("\n--- TEST 6: FP32 backbone → FP32 Dense → bf16 norm (isolate L2 norm) ---")
    model = models.ColBERT(model_name_or_path=MODEL_DIR, model_kwargs={"torch_dtype": torch.float32}, device="cuda")
    model.eval()

    def hook_cast_to_bf16_after(features, orig_fwd):
        features = orig_fwd(features)
        features["token_embeddings"] = features["token_embeddings"].bfloat16()
        return features

    d, q = encode_with_hook(model, doc_inputs, queries, hook_cast_to_bf16_after)
    compare("fp32_backbone→fp32_dense→bf16_norm", d, ref_doc, "doc")
    compare("fp32_backbone→fp32_dense→bf16_norm", q, ref_q, "query")
    del model; torch.cuda.empty_cache()

    # === TEST 7: Decomposed stage-by-stage ===
    print("\n--- TEST 7: Stage-by-stage error decomposition ---")
    intermediates = {}

    for dtype_label, dtype_arg in [("fp32", torch.float32), ("bf16", None)]:
        kwargs = {"model_kwargs": {"torch_dtype": dtype_arg}} if dtype_arg else {}
        m = models.ColBERT(model_name_or_path=MODEL_DIR, device="cuda", **kwargs)
        m.eval()

        captured = {}
        orig = m[1].forward
        def make_capture(cap, orig_fn):
            def hook(features):
                cap["backbone_out"] = features["token_embeddings"].detach().cpu().clone()
                features = orig_fn(features)
                cap["dense_out"] = features["token_embeddings"].detach().cpu().clone()
                return features
            return hook
        m[1].forward = make_capture(captured, orig)

        with torch.no_grad():
            m.encode(doc_inputs, is_query=False, batch_size=1)
        intermediates[dtype_label] = captured
        del m; torch.cuda.empty_cache()

    print("\n  Stage-by-stage bf16 vs fp32 (document encoding):")
    for stage in ["backbone_out", "dense_out"]:
        bf16_t = intermediates["bf16"][stage].float()
        fp32_t = intermediates["fp32"][stage].float()
        min_seq = min(bf16_t.shape[1], fp32_t.shape[1])
        bf16_t = bf16_t[:, :min_seq, :]
        fp32_t = fp32_t[:, :min_seq, :]

        delta = (bf16_t - fp32_t).abs()
        rel_err = delta / fp32_t.abs().clamp(min=1e-10)
        cos = F.cosine_similarity(bf16_t.reshape(-1, bf16_t.shape[-1]), fp32_t.reshape(-1, fp32_t.shape[-1]), dim=1)

        print(f"    {stage:20s} shape={list(bf16_t.shape)} dim={bf16_t.shape[-1]}")
        print(f"      abs error:  max={delta.max().item():.4e}  mean={delta.mean().item():.4e}")
        print(f"      rel error:  max={rel_err.max().item():.4e}  mean={rel_err.mean().item():.4e}")
        print(f"      cosine sim: min={cos.min().item():.8f}  mean={cos.mean().item():.8f}")

    # L2 normalize both and compare
    bf16_dense = intermediates["bf16"]["dense_out"].float()
    fp32_dense = intermediates["fp32"]["dense_out"].float()
    min_seq = min(bf16_dense.shape[1], fp32_dense.shape[1])
    bf16_normed = F.normalize(bf16_dense[:, :min_seq, :], p=2, dim=2)
    fp32_normed = F.normalize(fp32_dense[:, :min_seq, :], p=2, dim=2)
    delta_normed = (bf16_normed - fp32_normed).abs()
    cos_normed = F.cosine_similarity(bf16_normed.reshape(-1, 128), fp32_normed.reshape(-1, 128), dim=1)
    print(f"    {'after_L2_norm':20s} shape={list(bf16_normed.shape)} dim=128")
    print(f"      abs error:  max={delta_normed.max().item():.4e}  mean={delta_normed.mean().item():.4e}")
    print(f"      cosine sim: min={cos_normed.min().item():.8f}  mean={cos_normed.mean().item():.8f}")

    # Error amplification
    backbone_err = (intermediates["bf16"]["backbone_out"].float()[:, :min_seq, :] - intermediates["fp32"]["backbone_out"].float()[:, :min_seq, :]).abs().mean().item()
    dense_err = (bf16_dense[:, :min_seq, :] - fp32_dense[:, :min_seq, :]).abs().mean().item()
    norm_err = delta_normed.mean().item()
    print(f"\n    Error propagation:")
    print(f"      backbone mean abs err: {backbone_err:.4e}")
    print(f"      dense mean abs err:    {dense_err:.4e} ({dense_err/backbone_err:.1f}x backbone)")
    print(f"      after norm mean abs:   {norm_err:.4e} ({norm_err/dense_err:.1f}x dense)")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("  Test 1: Full bf16 error (the bug)")
    print("  Test 2: Error if we only fix Dense+norm → shows backbone contribution")
    print("  Test 3: Error if we only fix norm → shows backbone+Dense contribution")
    print("  Test 4: Error if only backbone is fp32 → shows Dense+norm contribution")
    print("  Test 5: Error from bf16 Dense alone (fp32 backbone, fp32 norm)")
    print("  Test 6: Error from bf16 norm alone (fp32 backbone, fp32 Dense)")


if __name__ == "__main__":
    main()
