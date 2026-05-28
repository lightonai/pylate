"""CUDA-gated parity and training-smoke tests for the LIK MaxSim integration.

These tests verify that the fused kernel agrees with the einsum reference on
both forward and backward, that the KD scoring path matches, and that
``colbert_scores`` trains without producing NaNs / collapsing. They are
marked ``@pytest.mark.slow`` so they are skipped by the default CPU CI; run
them with ``pytest -m slow`` on a host with a CUDA Ampere+ GPU and
``late-interaction-kernels`` installed.
"""

import pytest
import torch

from pylate.scores import colbert_kd_scores, colbert_scores
from pylate.utils.maxsim import _dispatch_path, _torch_maxsim, maxsim_inbatch

pytest.importorskip("late_interaction_kernels")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        reason="LIK kernel requires Ampere or newer",
    ),
]

EMBEDDING_DIM: int = 128
BATCH_SIZE: int = 4
QUERY_LEN: int = 16
DOC_LEN: int = 32

_DTYPES: list[torch.dtype] = [torch.float32, torch.float16, torch.bfloat16]

# Tolerances calibrated to match the LIK project's own tests
# (see late-interaction-kernels/tests/test_padded.py).
_FORWARD_TOL: dict[torch.dtype, float] = {
    torch.float32: 5e-3,
    torch.float16: 2e-2,
    torch.bfloat16: 2e-2,
}
_BACKWARD_TOL: dict[torch.dtype, float] = {
    torch.float32: 1e-2,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


def _random_inputs(
    dtype: torch.dtype, requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    query: torch.Tensor = torch.randn(
        BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda"
    )
    doc: torch.Tensor = torch.randn(
        BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda"
    )
    if requires_grad:
        query.requires_grad_(True)
        doc.requires_grad_(True)
    return query, doc


def test_dispatch_path_returns_cuda() -> None:
    query, doc = _random_inputs(torch.float32)
    assert _dispatch_path(query, doc) == "cuda"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_parity_cuda(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, doc = _random_inputs(dtype)

    got: torch.Tensor = maxsim_inbatch(query, doc)

    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    want: torch.Tensor = maxsim_inbatch(query, doc)
    # Sanity: the disabled path equals the einsum reference exactly.
    assert torch.equal(want, _torch_maxsim(query, doc, None, None))

    assert got.shape == want.shape
    tol: float = _FORWARD_TOL[dtype]
    # LIK always returns fp32; cast the torch reference to fp32 too so the
    # comparison isn't bottlenecked by bf16/fp16 accumulation noise.
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_parity_cuda(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LIK side: input dtype as-is, the kernel accumulates in fp32 internally.
    query_lik, doc_lik = _random_inputs(dtype, requires_grad=True)
    maxsim_inbatch(query_lik, doc_lik).sum().backward()

    # Reference side: cast the same input values to fp32 so the torch reduce
    # runs in fp32 too. Comparing against a bf16/fp16 torch reduce instead
    # would surface accumulation noise — not a kernel discrepancy.
    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    query_ref_dtype, doc_ref_dtype = _random_inputs(dtype)
    query_ref: torch.Tensor = query_ref_dtype.float().requires_grad_(True)
    doc_ref: torch.Tensor = doc_ref_dtype.float().requires_grad_(True)
    maxsim_inbatch(query_ref, doc_ref).sum().backward()

    tol: float = _BACKWARD_TOL[dtype]
    assert query_lik.grad is not None and query_ref.grad is not None
    assert doc_lik.grad is not None and doc_ref.grad is not None
    torch.testing.assert_close(
        query_lik.grad.float(), query_ref.grad, rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        doc_lik.grad.float(), doc_ref.grad, rtol=tol, atol=tol
    )


def test_colbert_scores_with_masks_parity_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end public-surface check with non-trivial masks."""
    torch.manual_seed(1)
    query: torch.Tensor = torch.randn(3, 8, EMBEDDING_DIM, device="cuda")
    doc: torch.Tensor = torch.randn(5, 12, EMBEDDING_DIM, device="cuda")
    query_mask: torch.Tensor = torch.tensor(
        [[1.0] * 6 + [0.0] * 2, [1.0] * 8, [1.0] * 4 + [0.0] * 4], device="cuda"
    )
    doc_mask: torch.Tensor = torch.ones(5, 12, device="cuda")
    doc_mask[:, 10:] = 0.0

    got: torch.Tensor = colbert_scores(query, doc, query_mask, doc_mask)

    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    want: torch.Tensor = colbert_scores(query, doc, query_mask, doc_mask)

    torch.testing.assert_close(got.float(), want.float(), rtol=5e-3, atol=5e-3)


def test_colbert_kd_scores_parity_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """KD scoring (per-query candidate lists) must agree with the einsum reference."""
    torch.manual_seed(2)
    num_queries, num_docs = 3, 4
    query: torch.Tensor = torch.randn(num_queries, 6, EMBEDDING_DIM, device="cuda")
    doc: torch.Tensor = torch.randn(
        num_queries, num_docs, 10, EMBEDDING_DIM, device="cuda"
    )
    query_mask: torch.Tensor = torch.ones(num_queries, 6, device="cuda")
    doc_mask: torch.Tensor = torch.ones(num_queries, num_docs, 10, device="cuda")

    got: torch.Tensor = colbert_kd_scores(query, doc, query_mask, doc_mask)

    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    want: torch.Tensor = colbert_kd_scores(query, doc, query_mask, doc_mask)

    torch.testing.assert_close(got.float(), want.float(), rtol=5e-3, atol=5e-3)


def test_colbert_kd_scores_variable_lengths_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KD parity with non-trivial right-padded lengths derived from the masks."""
    torch.manual_seed(3)
    num_queries, num_docs = 4, 5
    query_len, doc_len = 12, 24
    query: torch.Tensor = torch.randn(num_queries, query_len, EMBEDDING_DIM, device="cuda")
    doc: torch.Tensor = torch.randn(
        num_queries, num_docs, doc_len, EMBEDDING_DIM, device="cuda"
    )
    # Right-padded contiguous-prefix masks (HF tokenizer convention).
    query_mask: torch.Tensor = torch.zeros(num_queries, query_len, device="cuda")
    for query_index in range(num_queries):
        query_mask[query_index, : 6 + query_index] = 1.0
    doc_mask: torch.Tensor = torch.zeros(num_queries, num_docs, doc_len, device="cuda")
    for query_index in range(num_queries):
        for doc_index in range(num_docs):
            doc_mask[query_index, doc_index, : 10 + doc_index] = 1.0

    got: torch.Tensor = colbert_kd_scores(query, doc, query_mask, doc_mask)

    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    want: torch.Tensor = colbert_kd_scores(query, doc, query_mask, doc_mask)

    torch.testing.assert_close(got.float(), want.float(), rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_colbert_kd_scores_backward_parity_cuda(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KD gradients from the LIK path must match an fp32 einsum reference."""
    torch.manual_seed(4)
    num_queries, num_docs, query_len, doc_len = 3, 4, 8, 16

    def _build(requires_grad: bool, *, cast_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # Re-seed inside the helper so both calls produce identical input values
        # regardless of the dtype we cast to.
        torch.manual_seed(4)
        query: torch.Tensor = torch.randn(
            num_queries, query_len, EMBEDDING_DIM, device="cuda", dtype=cast_dtype
        )
        doc: torch.Tensor = torch.randn(
            num_queries, num_docs, doc_len, EMBEDDING_DIM, device="cuda", dtype=cast_dtype
        )
        if requires_grad:
            query.requires_grad_(True)
            doc.requires_grad_(True)
        return query, doc

    # LIK path at the input dtype (kernel accumulates in fp32 internally).
    query_lik, doc_lik = _build(requires_grad=True, cast_dtype=dtype)
    colbert_kd_scores(query_lik, doc_lik).sum().backward()

    # Reference path in fp32 — same input values cast up, einsum runs in fp32.
    monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
    query_ref, doc_ref = _build(requires_grad=True, cast_dtype=torch.float32)
    colbert_kd_scores(query_ref, doc_ref).sum().backward()

    tol: float = _BACKWARD_TOL[dtype]
    assert query_lik.grad is not None and query_ref.grad is not None
    assert doc_lik.grad is not None and doc_ref.grad is not None
    torch.testing.assert_close(
        query_lik.grad.float(), query_ref.grad, rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        doc_lik.grad.float(), doc_ref.grad, rtol=tol, atol=tol
    )


def test_training_smoke_cuda() -> None:
    """5 SGD steps through ``colbert_scores``; loss must stay finite and trend down."""
    torch.manual_seed(0)
    query: torch.Tensor = torch.randn(
        BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True
    )
    doc: torch.Tensor = torch.randn(
        BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True
    )
    optimizer = torch.optim.SGD([query, doc], lr=1e-2)
    labels: torch.Tensor = torch.arange(BATCH_SIZE, device="cuda")

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        # Cross-entropy over [B, B] scores: positives on the diagonal.
        scores: torch.Tensor = colbert_scores(query, doc)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    assert losses[-1] <= losses[0], f"loss did not decrease: {losses}"
