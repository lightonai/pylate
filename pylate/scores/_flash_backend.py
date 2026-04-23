"""Optional flash-maxsim backend for ColBERT scoring.

This module is imported lazily by `pylate.scores.scores` when the user selects
the `flash` backend (or `auto` and the inputs/environment support it). It
provides drop-in equivalents for the three public scoring functions, handling
the mask <-> lengths conversion and the shared-docs / per-query-docs branch.

Requires: `pip install flash-maxsim>=0.2.1`
"""

from __future__ import annotations

from typing import Optional

import torch

_IMPORT_OK = None  # tri-state: None = not checked yet, True/False after first check


def is_available() -> bool:
    """Return True if flash-maxsim is importable and CUDA is available."""
    global _IMPORT_OK
    if _IMPORT_OK is not None:
        return _IMPORT_OK
    try:
        import flash_maxsim  # noqa: F401

        _IMPORT_OK = torch.cuda.is_available()
    except ImportError:
        _IMPORT_OK = False
    return _IMPORT_OK


def _mask_to_lengths(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert a [..., L] 0/1 mask to a [...] int32 length tensor. Returns
    None if `mask` is None."""
    if mask is None:
        return None
    # mask is typically FP16/FP32 with 0.0/1.0 values; sum along last dim.
    return mask.sum(dim=-1).to(torch.int32).contiguous()


def _inputs_supported(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Conservative gate: only use flash when inputs are CUDA + ≥16 tokens.

    Flash kernels use tensor cores with ≥16-element tiles; very small shapes
    are faster on pytorch's cuBLAS path. Also require dtype in {fp16, bf16,
    fp32} (flash casts to fp16 internally).
    """
    if not (Q.is_cuda and D.is_cuda):
        return False
    # Reject pathological shapes where CSR-overhead or launch cost dominates.
    if Q.numel() == 0 or D.numel() == 0:
        return False
    return True


def colbert_scores_flash(
    queries_embeddings: torch.Tensor,  # [Nq, Lq, d]
    documents_embeddings: torch.Tensor,  # [B, Ld, d]
    queries_mask: Optional[torch.Tensor] = None,
    documents_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`colbert_scores` via flash-maxsim. Returns [Nq, B] scores."""
    if not _inputs_supported(queries_embeddings, documents_embeddings):
        raise RuntimeError("flash backend not applicable at this shape")

    needs_grad = queries_embeddings.requires_grad or documents_embeddings.requires_grad

    q_lens = _mask_to_lengths(queries_mask)
    d_lens = _mask_to_lengths(documents_mask)

    if needs_grad:
        from flash_maxsim import flash_maxsim_batched_train

        return flash_maxsim_batched_train(
            queries_embeddings,
            documents_embeddings,
            shared_docs=True,
            doc_lengths=d_lens,
            query_lengths=q_lens,
        )

    # No-grad path. flash_maxsim_batched's forward kernel does not mask padded
    # query positions from query_lengths, so we pre-zero them to match pylate's
    # mul-by-mask semantics.
    Q = queries_embeddings
    if queries_mask is not None:
        Q = Q * queries_mask.unsqueeze(-1).to(Q.dtype)
    from flash_maxsim import flash_maxsim_batched

    return flash_maxsim_batched(
        Q,
        documents_embeddings,
        doc_lengths=d_lens,
        shared_docs=True,
        query_lengths=q_lens,
    )


def colbert_scores_pairwise_flash(
    queries_embeddings: torch.Tensor,  # [B, Lq, d] padded
    documents_embeddings: torch.Tensor,  # [B, Ld, d] padded
) -> torch.Tensor:
    """`colbert_scores_pairwise` via flash-maxsim varlen — replaces the
    Python `for` loop with a single fused kernel."""
    if not _inputs_supported(queries_embeddings, documents_embeddings):
        raise RuntimeError("flash backend not applicable at this shape")

    from flash_maxsim import flash_maxsim_varlen, pack_pairs

    # pack_pairs expects list[Tensor] of per-pair shapes. Convert padded batch
    # to list. This allocates but avoids the slow Python loop pylate uses.
    B = queries_embeddings.shape[0]
    q_list = [queries_embeddings[i] for i in range(B)]
    d_list = [documents_embeddings[i] for i in range(B)]
    Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(q_list, d_list)
    return flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md)


def colbert_kd_scores_flash(
    queries_embeddings: torch.Tensor,  # [Nq, Lq, d]
    documents_embeddings: torch.Tensor,  # [Nq, B, Ld, d]  — per-query teachers
    queries_mask: Optional[torch.Tensor] = None,
    documents_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`colbert_kd_scores` via flash-maxsim (shared_docs=False)."""
    if not _inputs_supported(queries_embeddings, documents_embeddings):
        raise RuntimeError("flash backend not applicable at this shape")

    needs_grad = queries_embeddings.requires_grad or documents_embeddings.requires_grad

    q_lens = _mask_to_lengths(queries_mask)
    # documents_mask is [Nq, B, Ld] in KD path — flatten to [Nq*B] lengths.
    d_lens = None
    if documents_mask is not None:
        Nq, B, Ld = documents_mask.shape
        d_lens = documents_mask.sum(dim=-1).reshape(Nq * B).to(torch.int32).contiguous()

    if needs_grad:
        from flash_maxsim import flash_maxsim_batched_train

        return flash_maxsim_batched_train(
            queries_embeddings,
            documents_embeddings,
            shared_docs=False,
            doc_lengths=d_lens,
            query_lengths=q_lens,
        )
    else:
        from flash_maxsim import flash_maxsim_batched_train

        with torch.no_grad():
            return flash_maxsim_batched_train(
                queries_embeddings,
                documents_embeddings,
                shared_docs=False,
                doc_lengths=d_lens,
                query_lengths=q_lens,
            )
