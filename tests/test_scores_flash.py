"""Backend-parity tests for pylate.scores — torch vs flash-maxsim.

Skipped when CUDA or `flash_maxsim` is unavailable so the suite still passes
on CPU-only CI.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from pylate.scores import (
    colbert_kd_scores,
    colbert_scores,
    colbert_scores_pairwise,
)

_FLASH_INSTALLED = importlib.util.find_spec("flash_maxsim") is not None
_HAS_CUDA = torch.cuda.is_available()

requires_flash = pytest.mark.skipif(
    not (_FLASH_INSTALLED and _HAS_CUDA),
    reason="requires CUDA + flash-maxsim",
)


def _norm(shape, dtype=torch.float16, device="cuda"):
    x = torch.randn(*shape, dtype=dtype, device=device)
    return F.normalize(x, dim=-1)


@requires_flash
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_colbert_scores_parity(dtype):
    torch.manual_seed(0)
    Nq, B, Lq, Ld, d = 4, 8, 32, 180, 128
    Q = _norm((Nq, Lq, d), dtype=dtype)
    D = _norm((B, Ld, d), dtype=dtype)
    q_mask = torch.ones(Nq, Lq, device="cuda", dtype=dtype)
    d_mask = torch.ones(B, Ld, device="cuda", dtype=dtype)
    # vary a few rows to exercise masking
    q_mask[1, 20:] = 0
    d_mask[3, 100:] = 0

    ref = colbert_scores(Q, D, q_mask, d_mask, backend="torch")
    fast = colbert_scores(Q, D, q_mask, d_mask, backend="flash")

    atol = 5e-2 if dtype == torch.float16 else 1e-3
    torch.testing.assert_close(fast.float(), ref.float(), atol=atol, rtol=atol)


@requires_flash
def test_colbert_scores_pairwise_parity():
    torch.manual_seed(0)
    B, Lq, Ld, d = 6, 32, 180, 128
    Q = _norm((B, Lq, d))
    D = _norm((B, Ld, d))

    ref = colbert_scores_pairwise(Q, D, backend="torch")
    fast = colbert_scores_pairwise(Q, D, backend="flash")

    torch.testing.assert_close(fast.float(), ref.float(), atol=5e-2, rtol=5e-2)


@requires_flash
def test_colbert_kd_scores_parity():
    torch.manual_seed(0)
    Nq, B, Lq, Ld, d = 4, 4, 32, 180, 128
    Q = _norm((Nq, Lq, d))
    D = _norm((Nq, B, Ld, d))
    q_mask = torch.ones(Nq, Lq, device="cuda", dtype=Q.dtype)
    d_mask = torch.ones(Nq, B, Ld, device="cuda", dtype=Q.dtype)
    q_mask[0, 28:] = 0
    d_mask[1, 2, 150:] = 0

    ref = colbert_kd_scores(Q, D, q_mask, d_mask, backend="torch")
    fast = colbert_kd_scores(Q, D, q_mask, d_mask, backend="flash")

    torch.testing.assert_close(fast.float(), ref.float(), atol=5e-2, rtol=5e-2)


@requires_flash
def test_colbert_scores_grad_parity():
    """Backward pass must match the torch path on cos-sim of grads."""
    torch.manual_seed(0)
    Nq, B, Lq, Ld, d = 4, 8, 32, 180, 128
    Q0 = _norm((Nq, Lq, d)).detach()
    D0 = _norm((B, Ld, d)).detach()

    def run(backend: str):
        Q = Q0.clone().requires_grad_(True)
        D = D0.clone().requires_grad_(True)
        scores = colbert_scores(Q, D, backend=backend)
        scores.diagonal().sum().backward()
        return scores.detach(), Q.grad.detach(), D.grad.detach()

    s_ref, gQ_ref, gD_ref = run("torch")
    s_f, gQ_f, gD_f = run("flash")

    torch.testing.assert_close(s_f.float(), s_ref.float(), atol=5e-2, rtol=5e-2)

    def cos(a, b):
        return F.cosine_similarity(
            a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
        ).item()

    assert cos(gQ_f, gQ_ref) > 0.99, f"grad_Q cos={cos(gQ_f, gQ_ref)}"
    assert cos(gD_f, gD_ref) > 0.99, f"grad_D cos={cos(gD_f, gD_ref)}"


def test_backend_arg_accepted_cpu():
    """Passing backend='auto' on CPU must be a no-op (falls through to torch)."""
    Q = torch.tensor([[[1.0], [0.0]]])
    D = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    scores = colbert_scores(Q, D, backend="auto")
    assert scores.shape == (1, 2)


def test_backend_invalid():
    with pytest.raises(ValueError):
        colbert_scores(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), backend="nope")


def test_flash_backend_module_imports():
    """The lazy-imported backend module must import cleanly even without CUDA
    or flash-maxsim installed (CPU CI environment)."""
    from pylate.scores import _flash_backend

    assert callable(_flash_backend.is_available)
    # is_available() is False without CUDA — don't assert on it since CI may vary,
    # but calling it must not raise.
    _flash_backend.is_available()


def test_backend_flash_without_cuda_raises():
    """Explicit backend='flash' must raise on CPU tensors (no silent fallback)."""
    with pytest.raises(RuntimeError, match="requires CUDA"):
        colbert_scores(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), backend="flash")
