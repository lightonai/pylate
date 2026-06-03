"""Backend-parity tests for pylate.scores — torch vs late-interaction-kernels.

The ``@requires_lik`` parity tests are skipped when CUDA or
``late_interaction_kernels`` is unavailable, so the suite still passes on
CPU-only CI. The gating tests below exercise the pure-Python dispatch logic
and run everywhere.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from pylate.scores import (
    _lik_backend,
    colbert_kd_scores,
    colbert_scores,
    colbert_scores_pairwise,
)
from pylate.scores._lik_backend import LIKUnsupported

_LIK_INSTALLED = importlib.util.find_spec("late_interaction_kernels") is not None
_HAS_CUDA = torch.cuda.is_available()

requires_lik = pytest.mark.skipif(
    not (_LIK_INSTALLED and _HAS_CUDA),
    reason="requires CUDA + late-interaction-kernels",
)

EMBEDDING_DIM: int = 128


def _norm(shape: tuple[int, ...], dtype: torch.dtype = torch.float16) -> torch.Tensor:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Dispatch gating — pure Python, runs on CPU CI.
# ---------------------------------------------------------------------------


def test_backend_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError):
        colbert_scores(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8), backend="nope")


def test_backend_lik_without_accelerator_raises() -> None:
    """Explicit backend='lik' must raise on CPU tensors (no silent fallback)."""
    with pytest.raises(RuntimeError, match="CUDA or MPS"):
        colbert_scores(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8), backend="lik")


def test_backend_auto_noops_on_cpu() -> None:
    """backend='auto' on CPU falls through to the torch path."""
    query = torch.tensor([[[1.0], [0.0]]])
    doc = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    scores = colbert_scores(query, doc, backend="auto")
    assert scores.shape == (1, 2)


def test_lik_backend_module_imports() -> None:
    """The lazily-imported backend module imports cleanly without CUDA/LIK and
    ``is_available()`` is callable (must not raise)."""
    assert callable(_lik_backend.is_available)
    _lik_backend.is_available()


@pytest.mark.parametrize("head_dim", [4, 100, 264])
def test_lik_device_rejects_bad_head_dim(head_dim: int) -> None:
    """head dim must be ≥ 8, a multiple of 8, and ≤ 256 (4: too small, 100:
    not a multiple of 8, 264: too large)."""
    query = torch.randn(2, 4, head_dim)
    doc = torch.randn(3, 5, head_dim)
    with pytest.raises(LIKUnsupported):
        _lik_backend._lik_device(query, doc)


def test_lik_device_rejects_unsupported_dtype() -> None:
    query = torch.randn(2, 4, EMBEDDING_DIM, dtype=torch.float64)
    doc = torch.randn(3, 5, EMBEDDING_DIM, dtype=torch.float64)
    with pytest.raises(LIKUnsupported):
        _lik_backend._lik_device(query, doc)


def test_lik_device_rejects_empty_tensor() -> None:
    query = torch.randn(0, 4, EMBEDDING_DIM)
    doc = torch.randn(3, 5, EMBEDDING_DIM)
    with pytest.raises(LIKUnsupported):
        _lik_backend._lik_device(query, doc)


def test_lik_device_rejects_cpu() -> None:
    """A well-shaped fp32 CPU tensor still has no accelerator to run on."""
    query = torch.randn(2, 4, EMBEDDING_DIM)
    doc = torch.randn(3, 5, EMBEDDING_DIM)
    with pytest.raises(LIKUnsupported):
        _lik_backend._lik_device(query, doc)


def test_mask_as_bool() -> None:
    assert _lik_backend._mask_as_bool(None) is None
    float_mask = torch.tensor([[1.0, 0.0, 1.0]])
    assert _lik_backend._mask_as_bool(float_mask).dtype == torch.bool
    bool_mask = torch.tensor([[True, False]])
    assert _lik_backend._mask_as_bool(bool_mask) is bool_mask


# ---------------------------------------------------------------------------
# Parity vs the torch path — CUDA + LIK only.
# ---------------------------------------------------------------------------


@requires_lik
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_colbert_scores_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    Nq, B, Lq, Ld = 4, 8, 32, 180
    query = _norm((Nq, Lq, EMBEDDING_DIM), dtype=dtype)
    doc = _norm((B, Ld, EMBEDDING_DIM), dtype=dtype)
    q_mask = torch.ones(Nq, Lq, device="cuda")
    d_mask = torch.ones(B, Ld, device="cuda")
    q_mask[1, 20:] = 0
    d_mask[3, 100:] = 0

    ref = colbert_scores(query, doc, q_mask, d_mask, backend="torch")
    got = colbert_scores(query, doc, q_mask, d_mask, backend="lik")

    atol = 5e-2 if dtype != torch.float32 else 5e-3
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=atol)


@requires_lik
def test_colbert_scores_pairwise_parity() -> None:
    torch.manual_seed(0)
    B, Lq, Ld = 6, 32, 180
    query = _norm((B, Lq, EMBEDDING_DIM))
    doc = _norm((B, Ld, EMBEDDING_DIM))

    ref = colbert_scores_pairwise(query, doc, backend="torch")
    got = colbert_scores_pairwise(query, doc, backend="lik")

    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=5e-2)


@requires_lik
def test_colbert_kd_scores_parity() -> None:
    torch.manual_seed(0)
    Nq, B, Lq, Ld = 4, 4, 32, 180
    query = _norm((Nq, Lq, EMBEDDING_DIM))
    doc = _norm((Nq, B, Ld, EMBEDDING_DIM))
    q_mask = torch.ones(Nq, Lq, device="cuda")
    d_mask = torch.ones(Nq, B, Ld, device="cuda")
    q_mask[0, 28:] = 0
    d_mask[1, 2, 150:] = 0

    ref = colbert_kd_scores(query, doc, q_mask, d_mask, backend="torch")
    got = colbert_kd_scores(query, doc, q_mask, d_mask, backend="lik")

    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=5e-2)


@requires_lik
def test_colbert_kd_scores_variable_lengths_parity() -> None:
    """KD parity with right-padded per-(query, doc) lengths."""
    torch.manual_seed(3)
    Nq, B, Lq, Ld = 4, 5, 12, 24
    query = _norm((Nq, Lq, EMBEDDING_DIM), dtype=torch.float32)
    doc = _norm((Nq, B, Ld, EMBEDDING_DIM), dtype=torch.float32)
    q_mask = torch.zeros(Nq, Lq, device="cuda")
    for i in range(Nq):
        q_mask[i, : 6 + i] = 1.0
    d_mask = torch.zeros(Nq, B, Ld, device="cuda")
    for i in range(Nq):
        for j in range(B):
            d_mask[i, j, : 10 + j] = 1.0

    ref = colbert_kd_scores(query, doc, q_mask, d_mask, backend="torch")
    got = colbert_kd_scores(query, doc, q_mask, d_mask, backend="lik")

    torch.testing.assert_close(got.float(), ref.float(), atol=5e-3, rtol=5e-3)


@requires_lik
def test_colbert_scores_grad_parity() -> None:
    """Backward must match the torch path on cos-sim of grad_Q / grad_D."""
    torch.manual_seed(0)
    Nq, B, Lq, Ld = 4, 8, 32, 180
    query0 = _norm((Nq, Lq, EMBEDDING_DIM), dtype=torch.float32).detach()
    doc0 = _norm((B, Ld, EMBEDDING_DIM), dtype=torch.float32).detach()

    def run(backend: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query0.clone().requires_grad_(True)
        doc = doc0.clone().requires_grad_(True)
        scores = colbert_scores(query, doc, backend=backend)
        scores.diagonal().sum().backward()
        return scores.detach(), query.grad.detach(), doc.grad.detach()

    s_ref, gq_ref, gd_ref = run("torch")
    s_got, gq_got, gd_got = run("lik")

    torch.testing.assert_close(s_got.float(), s_ref.float(), atol=5e-3, rtol=5e-3)

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(
            a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
        ).item()

    assert cos(gq_got, gq_ref) > 0.99, f"grad_Q cos={cos(gq_got, gq_ref)}"
    assert cos(gd_got, gd_ref) > 0.99, f"grad_D cos={cos(gd_got, gd_ref)}"


@requires_lik
def test_training_smoke() -> None:
    """5 SGD steps through the LIK path; loss stays finite and trends down."""
    torch.manual_seed(0)
    B, Lq, Ld = 8, 16, 32
    query = (
        _norm((B, Lq, EMBEDDING_DIM), dtype=torch.float32).detach().requires_grad_(True)
    )
    doc = (
        _norm((B, Ld, EMBEDDING_DIM), dtype=torch.float32).detach().requires_grad_(True)
    )
    optimizer = torch.optim.SGD([query, doc], lr=1e-2)
    labels = torch.arange(B, device="cuda")

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        scores = colbert_scores(query, doc, backend="lik")
        loss = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    assert losses[-1] <= losses[0], f"loss did not decrease: {losses}"


@requires_lik
def test_lik_unsupported_strict_propagates() -> None:
    """backend='lik' (strict) re-raises LIKUnsupported instead of falling back."""
    query = torch.zeros(
        2, 4, 100, device="cuda", dtype=torch.float16
    )  # head dim 100: not a multiple of 8
    doc = torch.zeros(3, 5, 100, device="cuda", dtype=torch.float16)
    with pytest.raises(LIKUnsupported):
        colbert_scores(query, doc, backend="lik")


@requires_lik
def test_lik_unsupported_auto_falls_back() -> None:
    """backend='auto' silently falls back to torch when LIK can't run the shape."""
    torch.manual_seed(1)
    query = _norm((2, 4, 100))  # head dim 100: not a multiple of 8
    doc = _norm((3, 5, 100))
    got = colbert_scores(query, doc, backend="auto")
    ref = colbert_scores(query, doc, backend="torch")
    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=5e-2)
