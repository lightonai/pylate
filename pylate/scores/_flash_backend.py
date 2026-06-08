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


class FlashUnsupported(Exception):
    """Raised when the flash backend cannot run on the given inputs and the
    caller should fall back to the native PyTorch path. Other exceptions
    (assertion failures, real bugs in the kernel, OOMs, etc.) are *not*
    caught and will propagate to the user.

    Reasons: non-CUDA tensors, empty tensors, the flash-maxsim package not
    installed, or any other input-validation failure inside this backend.
    """


def _next_pow2(n: int) -> int:
    """Smallest power of two at or above ``n`` (``n``-clamped to ``>=1``)."""
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _bucket_lq(
    Q: torch.Tensor,
    q_lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pad ``Q``'s ``L_q`` dimension up to the next power of two so distinct
    upstream lengths reuse the same Triton autotune entry (avoids per-step
    recompilation when query lengths vary across batches; #212 review thread).

    Mathematically a no-op: a zero query row contributes
    ``max_j <0, D_j> = 0`` to the sum-of-maxes, and zero gradient to
    :math:`D`. If ``q_lengths`` is provided, it still points inside the
    original :math:`L_q` range and the kernel's per-query masking is
    preserved unchanged.
    """
    Lq = Q.shape[-2]
    Lq_pad = _next_pow2(Lq)
    if Lq_pad == Lq:
        return Q, q_lengths
    pad = Lq_pad - Lq
    # F.pad pads from the last dimension: (left, right, top, bottom). We want
    # to pad along L_q (second-to-last), bottom only.
    Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, pad))
    return Q_padded, q_lengths


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
    # mask is 0/1 but may be bf16/fp16: summing in that dtype rounds wrong
    # (bf16 represents integers exactly only up to 256, so a 365-token row
    # sums to 364). Force exact integer accumulation with dtype=int64.
    return mask.sum(dim=-1, dtype=torch.int64).to(torch.int32).contiguous()


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
        raise FlashUnsupported("flash backend not applicable at this shape")

    needs_grad = queries_embeddings.requires_grad or documents_embeddings.requires_grad

    q_lens = _mask_to_lengths(queries_mask)
    d_lens = _mask_to_lengths(documents_mask)

    if needs_grad:
        from flash_maxsim import flash_maxsim_batched_train

        Q_bucketed, q_lens_bucketed = _bucket_lq(queries_embeddings, q_lens)
        return flash_maxsim_batched_train(
            Q_bucketed,
            documents_embeddings,
            shared_docs=True,
            doc_lengths=d_lens,
            query_lengths=q_lens_bucketed,
        )

    # No-grad path. flash_maxsim_batched's forward kernel does not mask padded
    # query positions from query_lengths, so we pre-zero them to match pylate's
    # mul-by-mask semantics.
    Q = queries_embeddings
    if queries_mask is not None:
        Q = Q * queries_mask.unsqueeze(-1).to(Q.dtype)
    Q_bucketed, q_lens_bucketed = _bucket_lq(Q, q_lens)
    from flash_maxsim import flash_maxsim_batched

    return flash_maxsim_batched(
        Q_bucketed,
        documents_embeddings,
        doc_lengths=d_lens,
        shared_docs=True,
        query_lengths=q_lens_bucketed,
    )


def colbert_scores_pairwise_flash(
    queries_embeddings: torch.Tensor,  # [B, Lq, d] padded
    documents_embeddings: torch.Tensor,  # [B, Ld, d] padded
) -> torch.Tensor:
    """`colbert_scores_pairwise` via flash-maxsim varlen — replaces the
    Python `for` loop with a single fused kernel."""
    if not _inputs_supported(queries_embeddings, documents_embeddings):
        raise FlashUnsupported("flash backend not applicable at this shape")

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
        raise FlashUnsupported("flash backend not applicable at this shape")

    needs_grad = queries_embeddings.requires_grad or documents_embeddings.requires_grad

    q_lens = _mask_to_lengths(queries_mask)
    # documents_mask is [Nq, B, Ld] in KD path — flatten to [Nq*B] lengths.
    d_lens = None
    if documents_mask is not None:
        Nq, B, Ld = documents_mask.shape
        d_lens = documents_mask.sum(dim=-1).reshape(Nq * B).to(torch.int32).contiguous()

    Q_bucketed, q_lens_bucketed = _bucket_lq(queries_embeddings, q_lens)

    if needs_grad:
        from flash_maxsim import flash_maxsim_batched_train

        return flash_maxsim_batched_train(
            Q_bucketed,
            documents_embeddings,
            shared_docs=False,
            doc_lengths=d_lens,
            query_lengths=q_lens_bucketed,
        )
    else:
        from flash_maxsim import flash_maxsim_batched_train

        with torch.no_grad():
            return flash_maxsim_batched_train(
                Q_bucketed,
                documents_embeddings,
                shared_docs=False,
                doc_lengths=d_lens,
                query_lengths=q_lens_bucketed,
            )
