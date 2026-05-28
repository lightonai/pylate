"""MaxSim dispatch: fused LIK kernels when available, einsum reference otherwise.

Set ``PYLATE_DISABLE_LIK=1`` (or legacy ``LIK_DISABLE=1``) to force the reference path.
"""

import os

import torch

try:
    import late_interaction_kernels as _lik  # noqa: F401

    _LIK_AVAILABLE: bool = True
except ImportError:
    _LIK_AVAILABLE = False


# LIK's Triton kernels run on fp16/bf16/fp32 only; head dim must be a multiple
# of 8 (MMA tile constraint) and fit in shared memory (`d <= 256`).
_KERNEL_MAX_HEAD_DIM: int = 256
_KERNEL_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)


def _is_disabled() -> bool:
    """True when the user has set either kill-switch env var."""
    return (
        os.environ.get("PYLATE_DISABLE_LIK", "0") == "1"
        or os.environ.get("LIK_DISABLE", "0") == "1"
    )


def _mask_as_bool(mask: torch.Tensor | None) -> torch.Tensor | None:
    """pylate masks can be float (0/1), bool, or ``None``."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def _dispatch_path(query: torch.Tensor, doc: torch.Tensor) -> str | None:
    """Auto-select the fastest available MaxSim backend from device + shape heuristics.

    Returns ``"cuda"``, ``"mps"``, or ``None`` (fall back to ``_torch_maxsim``).
    """
    if not _LIK_AVAILABLE or _is_disabled():
        return None
    if query.device != doc.device:
        return None
    if query.dtype not in _KERNEL_SUPPORTED_DTYPES:
        return None
    # Triton MMA tiles require a head dim that is both ≥ 8 (lower bound) and
    # a multiple of 8 (tile alignment); shared-memory limits cap it at 256.
    head_dim: int = query.shape[-1]
    if head_dim < 8 or head_dim > _KERNEL_MAX_HEAD_DIM or head_dim % 8 != 0:
        return None
    if query.is_cuda and doc.is_cuda:
        # bf16 tensor cores require Ampere or newer
        if torch.cuda.get_device_capability(query.device)[0] < 8:
            return None
        return "cuda"
    if query.device.type == "mps" and doc.device.type == "mps":
        return "mps"
    return None


def _torch_maxsim(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None,
    doc_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference MaxSim: ``einsum("ash,bth->abst").max(-1).sum(-1)``."""
    scores: torch.Tensor = torch.einsum("ash,bth->abst", query, doc)
    if query_mask is not None:
        scores = scores * query_mask.unsqueeze(1).unsqueeze(3)
    if doc_mask is not None:
        scores = scores * doc_mask.unsqueeze(0).unsqueeze(2)
    return scores.max(axis=-1).values.sum(axis=-1)


def _torch_maxsim_kd(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None,
    doc_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference KD MaxSim: per-query candidate lists, ``doc`` is ``[Nq, Nd, Ld, d]``."""
    scores: torch.Tensor = torch.einsum("ash,abth->abst", query, doc)
    if query_mask is not None:
        scores = scores * query_mask.unsqueeze(1).unsqueeze(3)
    if doc_mask is not None:
        scores = scores * doc_mask.unsqueeze(2)
    return scores.max(axis=-1).values.sum(axis=-1)


def maxsim_inbatch(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch MaxSim scores for late-interaction scoring.

    Args:
        query: ``[Nq, Lq, d]`` query token embeddings.
        doc: ``[Nd, Ld, d]`` document token embeddings.
        query_mask: optional ``[Nq, Lq]`` boolean/float mask for query tokens.
        doc_mask: optional ``[Nd, Ld]`` boolean/float mask for doc tokens.

    Returns:
        ``[Nq, Nd]`` similarity matrix — the sum over query tokens of each
        token's max similarity against ``doc``'s token dimension.
    """
    path: str | None = _dispatch_path(query, doc)
    if path is None:
        return _torch_maxsim(query, doc, query_mask, doc_mask)

    q_mask: torch.Tensor | None = _mask_as_bool(query_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(doc_mask)

    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim as _lik_maxsim

        return _lik_maxsim(query, doc, q_mask=q_mask, d_mask=d_mask)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps as _lik_maxsim_mps

        return _lik_maxsim_mps(query, doc, q_mask=q_mask, d_mask=d_mask, normalize=False)
    return _torch_maxsim(query, doc, query_mask, doc_mask)


def maxsim_kd(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Knowledge-distillation MaxSim: each query has its own candidate list.

    Args:
        query: ``[Nq, Lq, d]`` query token embeddings.
        doc: ``[Nq, Nd, Ld, d]`` per-query document token embeddings.
        query_mask: optional ``[Nq, Lq]`` boolean/float mask. Arbitrary patterns
            supported (does not need to be a contiguous prefix).
        doc_mask: optional ``[Nq, Nd, Ld]`` mask, same convention.

    Returns:
        ``[Nq, Nd]`` similarity matrix.
    """
    path: str | None = _dispatch_path(query, doc)
    if path is None or doc.dim() != 4:
        return _torch_maxsim_kd(query, doc, query_mask, doc_mask)

    q_mask: torch.Tensor | None = _mask_as_bool(query_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(doc_mask)

    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim as _lik_maxsim

        # doc is [Nq, Nd, Ld, d]; the kernel detects the extra dim and takes
        # its kd_layout path — one fused launch, each program reads its own
        # K-slab.
        return _lik_maxsim(query, doc, q_mask=q_mask, d_mask=d_mask)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps as _lik_maxsim_mps

        # LIK's MPS kernel auto-detects 4-D doc and routes to the KD layout
        # (supported since late-interaction-kernels PR #79).
        return _lik_maxsim_mps(
            query, doc, q_mask=q_mask, d_mask=d_mask, normalize=False
        )
    return _torch_maxsim_kd(query, doc, query_mask, doc_mask)
