"""Optional late-interaction-kernels (LIK) backend for ColBERT scoring.

This module is imported lazily by ``pylate.scores.colbert`` when the user
selects the ``lik`` backend (or ``auto`` and the inputs/environment support
it). It provides drop-in equivalents for the three public scoring functions,
routing CUDA Ampere+ to LIK's fused Triton kernels and Apple Silicon to its
MPS kernels, and falls back (via :class:`LIKUnsupported`) to the caller's
pure-torch path otherwise.

Requires: ``pip install "pylate[lik]"`` (``late-interaction-kernels>=0.4.0``).
"""

import torch

# LIK's Triton kernels run on fp16/bf16/fp32 only; the head dim must be a
# multiple of 8 (MMA tile constraint) and fit in shared memory (``d <= 256``).
_KERNEL_MAX_HEAD_DIM: int = 256
_KERNEL_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)

# Tri-state import cache: None until first checked, then True/False.
_IMPORT_OK: bool | None = None


class LIKUnsupported(Exception):
    """Raised when the LIK backend cannot run on the given inputs and the
    caller should fall back to the pure-torch path.

    Only this exception triggers the silent ``auto`` fallback. Other errors
    (real kernel bugs, OOMs, assertion failures) propagate to the user.
    Reasons: non-CUDA/MPS tensors, mixed devices, unsupported dtype or head
    dim, empty tensors, or an old GPU below the kernel's compute capability.
    """


def is_available() -> bool:
    """True if ``late-interaction-kernels`` is importable and a supported
    accelerator (CUDA or Apple MPS) is present."""
    global _IMPORT_OK
    if _IMPORT_OK is not None:
        return _IMPORT_OK
    try:
        import late_interaction_kernels  # noqa: F401
    except ImportError:
        _IMPORT_OK = False
        return _IMPORT_OK
    _IMPORT_OK = torch.cuda.is_available() or torch.backends.mps.is_available()
    return _IMPORT_OK


def colbert_scores_lik(
    queries_embeddings: torch.Tensor,  # [Nq, Lq, d]
    documents_embeddings: torch.Tensor,  # [B, Ld, d]
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``colbert_scores`` via LIK. Returns ``[Nq, B]`` scores."""
    device: str = _lik_device(queries_embeddings, documents_embeddings)
    q_mask: torch.Tensor | None = _mask_as_bool(queries_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(documents_mask)

    if device == "mps":
        from late_interaction_kernels.mps import maxsim_mps

        return maxsim_mps(
            queries_embeddings,
            documents_embeddings,
            q_mask=q_mask,
            d_mask=d_mask,
            normalize=False,
        )

    from late_interaction_kernels.autograd import maxsim

    return maxsim(
        queries_embeddings, documents_embeddings, q_mask=q_mask, d_mask=d_mask
    )


def colbert_scores_pairwise_lik(
    queries_embeddings: torch.Tensor,  # [B, Lq, d]
    documents_embeddings: torch.Tensor,  # [B, Ld, d]
) -> torch.Tensor:
    """``colbert_scores_pairwise`` via LIK — replaces the Python ``for`` loop
    with a single fused diagonal kernel. Returns ``[B]``."""
    device: str = _lik_device(queries_embeddings, documents_embeddings)
    # maxsim_pairs is a CUDA Triton kernel with no MPS equivalent; let the
    # caller's torch loop handle Apple Silicon.
    if device != "cuda":
        raise LIKUnsupported("pairwise LIK path is CUDA-only")

    from late_interaction_kernels import maxsim_pairs

    return maxsim_pairs(queries_embeddings, documents_embeddings)


def colbert_kd_scores_lik(
    queries_embeddings: torch.Tensor,  # [Nq, Lq, d]
    documents_embeddings: torch.Tensor,  # [Nq, B, Ld, d] — per-query candidates
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``colbert_kd_scores`` via LIK. Returns ``[Nq, B]`` scores."""
    if documents_embeddings.dim() != 4:
        raise LIKUnsupported("KD path expects a 4-D documents tensor")

    device: str = _lik_device(queries_embeddings, documents_embeddings)
    # LIK's MPS kernel only handles the 3-D in-batch layout; the 4-D KD doc
    # has no MPS path, so defer Apple Silicon KD to the caller's torch loop.
    if device != "cuda":
        raise LIKUnsupported("KD layout is CUDA-only")

    q_mask: torch.Tensor | None = _mask_as_bool(queries_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(documents_mask)

    from late_interaction_kernels.autograd import maxsim

    # The 4-D doc makes maxsim take its kd_layout path: one fused launch, each
    # program reads its own K-slab — no packing.
    return maxsim(
        queries_embeddings, documents_embeddings, q_mask=q_mask, d_mask=d_mask
    )


def _mask_as_bool(mask: torch.Tensor | None) -> torch.Tensor | None:
    """pylate masks can be float (0/1), bool, or ``None``; LIK takes bool."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def _lik_device(query: torch.Tensor, doc: torch.Tensor) -> str:
    """Validate that LIK can run on these inputs and return the device kind
    (``"cuda"`` or ``"mps"``). Raise :class:`LIKUnsupported` otherwise."""
    if query.numel() == 0 or doc.numel() == 0:
        raise LIKUnsupported("empty input tensor")
    if query.device != doc.device:
        raise LIKUnsupported("queries and documents on different devices")
    if query.dtype not in _KERNEL_SUPPORTED_DTYPES:
        raise LIKUnsupported(f"unsupported dtype {query.dtype}")

    # Triton MMA tiles need a head dim ≥ 8 and a multiple of 8; shared memory
    # caps it at 256.
    head_dim: int = query.shape[-1]
    if head_dim < 8 or head_dim > _KERNEL_MAX_HEAD_DIM or head_dim % 8 != 0:
        raise LIKUnsupported(f"unsupported head dim {head_dim}")

    if query.is_cuda:
        # bf16 tensor cores require Ampere or newer.
        if torch.cuda.get_device_capability(query.device)[0] < 8:
            raise LIKUnsupported("LIK kernels require CUDA capability >= 8 (Ampere+)")
        return "cuda"
    if query.device.type == "mps":
        return "mps"
    raise LIKUnsupported("LIK backend requires CUDA or MPS inputs")
