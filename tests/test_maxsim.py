"""Tests for the late-interaction-kernels (LIK) dispatcher.

These tests exercise ``pylate/utils/maxsim.py``. The CPU paths always defer to
the einsum reference, so the bulk of the suite verifies the dispatcher's
gating behavior (env-var kill switches, device checks, embedding-dim floor)
and the end-to-end parity between the dispatcher and a kill-switched run.
The CUDA parity test is skipped unless a compatible GPU is present.
"""

import os
from typing import Iterator

import pytest
import torch

from pylate.scores import colbert_kd_scores, colbert_scores
from pylate.utils import maxsim as maxsim_module


@pytest.fixture
def clear_lik_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip the LIK kill-switch env vars for the duration of a test."""
    monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
    monkeypatch.delenv("LIK_DISABLE", raising=False)
    yield


class TestDispatcherGating:
    """``_dispatch_path`` must return None whenever the LIK kernels can't (or
    shouldn't) run."""

    def test_cpu_inputs_return_none(self, clear_lik_env: None) -> None:
        query = torch.randn(2, 4, 16)
        doc = torch.randn(3, 5, 16)
        assert maxsim_module._dispatch_path(query, doc) is None

    def test_lik_unavailable_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_lik_env: None,
    ) -> None:
        # Force the import-guard branch even when the package is installed, so
        # CPU CI catches regressions of the ``_LIK_AVAILABLE`` short-circuit.
        monkeypatch.setattr(maxsim_module, "_LIK_AVAILABLE", False)
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        query = torch.randn(2, 4, 16, device=device)
        doc = torch.randn(3, 5, 16, device=device)
        assert maxsim_module._dispatch_path(query, doc) is None

    def test_low_embedding_dim_returns_none(self, clear_lik_env: None) -> None:
        # H < 8 → reject, even on a supported device.
        query = torch.randn(2, 4, 4)
        doc = torch.randn(3, 5, 4)
        assert maxsim_module._dispatch_path(query, doc) is None

    def test_misaligned_embedding_dim_returns_none(self, clear_lik_env: None) -> None:
        # LIK's MMA tiles require head dim % 8 == 0 (e.g. d=72 fails).
        query = torch.randn(2, 4, 72)
        doc = torch.randn(3, 5, 72)
        assert maxsim_module._dispatch_path(query, doc) is None

    def test_oversized_embedding_dim_returns_none(self, clear_lik_env: None) -> None:
        # Shared-memory limits cap head dim at 256.
        query = torch.randn(2, 4, 384)
        doc = torch.randn(3, 5, 384)
        assert maxsim_module._dispatch_path(query, doc) is None

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int8])
    def test_unsupported_dtype_returns_none(
        self, dtype: torch.dtype, clear_lik_env: None
    ) -> None:
        # LIK supports fp16/bf16/fp32 only; anything else would raise inside
        # the kernel, so the dispatcher must fall back to einsum.
        query = torch.zeros(2, 4, 16, dtype=dtype)
        doc = torch.zeros(3, 5, 16, dtype=dtype)
        assert maxsim_module._dispatch_path(query, doc) is None

    def test_mixed_devices_return_none(self, clear_lik_env: None) -> None:
        query = torch.randn(2, 4, 16)
        meta_doc = torch.empty(3, 5, 16, device="meta")
        assert maxsim_module._dispatch_path(query, meta_doc) is None

    @pytest.mark.parametrize("flag", ["PYLATE_DISABLE_LIK", "LIK_DISABLE"])
    def test_kill_switch_disables_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        monkeypatch.setenv(flag, "1")
        query = torch.randn(2, 4, 16)
        doc = torch.randn(3, 5, 16)
        assert maxsim_module._dispatch_path(query, doc) is None


class TestMaskNormalization:
    """``_mask_as_bool`` accepts pylate's float / bool / None mask flavors."""

    def test_none_passthrough(self) -> None:
        assert maxsim_module._mask_as_bool(None) is None

    def test_bool_passthrough(self) -> None:
        mask = torch.tensor([[True, False, True]])
        out = maxsim_module._mask_as_bool(mask)
        assert out is mask
        assert out.dtype == torch.bool

    def test_float_converts_to_bool(self) -> None:
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        out = maxsim_module._mask_as_bool(mask)
        assert out.dtype == torch.bool
        assert out.tolist() == [[True, False, True]]


class TestEndToEndFallbackParity:
    """With LIK installed but CPU inputs, the dispatcher must take the
    reference path and produce identical results to a kill-switched run."""

    def test_colbert_scores_cpu_matches_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        query = torch.randn(2, 4, 16)
        doc = torch.randn(3, 5, 16)
        query_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        )
        doc_mask = torch.ones(3, 5)

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        with_dispatcher = colbert_scores(query, doc, query_mask, doc_mask)

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        forced_fallback = colbert_scores(query, doc, query_mask, doc_mask)

        torch.testing.assert_close(with_dispatcher, forced_fallback)

    def test_colbert_kd_scores_cpu_matches_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(1)
        num_queries, num_docs = 3, 4
        query = torch.randn(num_queries, 5, 16)
        doc = torch.randn(num_queries, num_docs, 7, 16)
        query_mask = torch.ones(num_queries, 5)
        doc_mask = torch.ones(num_queries, num_docs, 7)

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        with_dispatcher = colbert_kd_scores(query, doc, query_mask, doc_mask)

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        forced_fallback = colbert_kd_scores(query, doc, query_mask, doc_mask)

        torch.testing.assert_close(with_dispatcher, forced_fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCudaParity:
    """When CUDA is available the LIK Triton kernel must agree with the einsum
    reference within float tolerance. Skipped on CPU-only runners."""

    def test_colbert_scores_cuda_matches_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        query = torch.randn(4, 8, 32, device="cuda")
        doc = torch.randn(5, 12, 32, device="cuda")
        query_mask = torch.ones(4, 8, device="cuda")
        doc_mask = torch.ones(5, 12, device="cuda")

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        fused = colbert_scores(query, doc, query_mask, doc_mask)

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        reference = colbert_scores(query, doc, query_mask, doc_mask)

        torch.testing.assert_close(fused, reference, rtol=1e-3, atol=1e-3)
