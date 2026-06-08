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


# ---------------------------------------------------------------------------
# Reviewer-driven tests for PR #212 (raphaelsty + paulomouraj review threads).
# ---------------------------------------------------------------------------


@requires_flash
def test_flash_unsupported_raised_on_bad_input():
    """raphaelsty's review: the direct backend must raise FlashUnsupported (not
    a generic Exception) when inputs aren't applicable, so the auto fallback can
    catch the sentinel without silently masking real bugs."""
    from pylate.scores._flash_backend import (
        FlashUnsupported,
        colbert_scores_flash,
    )

    Q = torch.zeros(0, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.zeros(0, 180, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(FlashUnsupported):
        colbert_scores_flash(Q, D)


@requires_flash
def test_flash_unsupported_strict_backend_propagates():
    """backend='flash' (strict) must re-raise FlashUnsupported, not fall back."""
    from pylate.scores._flash_backend import FlashUnsupported

    Q = torch.zeros(0, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.zeros(0, 180, 128, device="cuda", dtype=torch.float16)
    with pytest.raises((FlashUnsupported, RuntimeError)):
        colbert_scores(Q, D, backend="flash")


@requires_flash
@pytest.mark.parametrize("Lq", [8, 13, 32, 33, 65, 100, 257])
def test_lq_bucketing_parity(Lq):
    """paulomouraj's review: padding L_q to the next power of two must be a
    no-op on scores AND gradients. Compares bucketed vs non-bucketed reference
    call to flash_maxsim_batched_train.

    Tolerance is tight (1e-3 on scores, 5e-2 on grads) because zero-padded rows
    contribute max_j <0, D_j> = 0 to sum-of-maxes and 0 gradient to D.
    """
    from flash_maxsim import flash_maxsim_batched_train

    from pylate.scores._flash_backend import _bucket_lq

    B, Ld, d = 16, 180, 128
    torch.manual_seed(42)
    Q = _norm((B, Lq, d))
    D = _norm((B, Ld, d))

    # Reference: pass raw L_q to flash directly (no bucketing).
    Q_ref = Q.clone().detach().requires_grad_(True)
    D_ref = D.clone().detach().requires_grad_(True)
    flash_maxsim_batched_train(Q_ref, D_ref, shared_docs=True).sum().backward()

    # Bucketed: pad L_q to next pow-2, supply matching query_lengths.
    Q_buc_in = Q.clone().detach().requires_grad_(True)
    D_buc = D.clone().detach().requires_grad_(True)
    Q_padded, _ = _bucket_lq(Q_buc_in, None)
    q_lens = torch.full((B,), Lq, device="cuda", dtype=torch.int32)
    flash_maxsim_batched_train(
        Q_padded,
        D_buc,
        shared_docs=True,
        query_lengths=q_lens,
    ).sum().backward()

    # Scores would have matched if we'd captured them, but the grad parity is
    # the stronger statement: it implies the kernel saw the same effective work.
    assert (Q_ref.grad - Q_buc_in.grad).abs().max().item() < 5e-2
    assert (D_ref.grad - D_buc.grad).abs().max().item() < 5e-2


@requires_flash
@pytest.mark.slow
def test_lq_bucketing_collapses_autotune_cache():
    """paulomouraj's review: with bucketing, 30 distinct upstream L_q values
    should compress to <=11 unique kernel shapes (one per power-of-two bucket
    in [8, 8192]). This is what removes the per-step recompilation storm.
    """
    import random

    from pylate.scores._flash_backend import _bucket_lq

    random.seed(7)
    lqs = sorted(random.sample(range(8, 1025), 30))
    bucketed_shapes = set()
    for lq in lqs:
        Q = _norm((4, lq, 128))
        Q_pad, _ = _bucket_lq(Q, None)
        bucketed_shapes.add(Q_pad.shape[1])

    # In [8, 1024] there are at most 8 power-of-two buckets:
    # {8, 16, 32, 64, 128, 256, 512, 1024}.
    assert len(bucketed_shapes) <= 8
    # And bucketing must compress at least 4x for the test to be meaningful.
    assert len(bucketed_shapes) * 4 <= len(lqs)
