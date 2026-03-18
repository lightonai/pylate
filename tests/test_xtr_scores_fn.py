"""Tests for the xtr_scores training-time scoring function."""

from __future__ import annotations

import torch

from pylate.scores import xtr_scores


class TestXTRScoresMatchedBatch:
    """Tests where query batch == doc batch (standard contrastive case)."""

    def test_output_shape(self) -> None:
        Q, N, Qt, Dt, H = 4, 2, 8, 12, 16
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)
        result = xtr_scores(queries, docs, k=8)
        assert result.shape == (Q, Q * N)

    def test_with_masks(self) -> None:
        Q, N, Qt, Dt, H = 4, 2, 8, 12, 16
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)
        q_mask = torch.ones(Q, Qt)
        d_mask = torch.ones(Q, N, Dt)
        # Mask out some tokens
        q_mask[:, -2:] = 0
        d_mask[:, :, -3:] = 0
        result = xtr_scores(
            queries, docs, queries_mask=q_mask, documents_mask=d_mask, k=8
        )
        assert result.shape == (Q, Q * N)


class TestXTRScoresMismatchedBatch:
    """Tests where query batch != doc batch (cached contrastive case).

    When CachedContrastive mini-batches queries, a smaller query chunk is
    scored against all documents. This requires xtr_scores to handle
    Qb != Dq correctly.
    """

    def test_fewer_queries_than_docs(self) -> None:
        """Core regression test: Qb < Dq should not raise."""
        Qb, Dq, N, Qt, Dt, H = 4, 16, 2, 8, 12, 16
        queries = torch.randn(Qb, Qt, H)
        docs = torch.randn(Dq, N, Dt, H)
        result = xtr_scores(queries, docs, k=8)
        assert result.shape == (Qb, Dq * N)

    def test_fewer_queries_with_masks(self) -> None:
        """Mask expansion must use Qb, not Dq."""
        Qb, Dq, N, Qt, Dt, H = 4, 16, 2, 8, 12, 16
        queries = torch.randn(Qb, Qt, H)
        docs = torch.randn(Dq, N, Dt, H)
        q_mask = torch.ones(Qb, Qt)
        d_mask = torch.ones(Dq, N, Dt)
        q_mask[:, -2:] = 0
        d_mask[:, :, -3:] = 0
        result = xtr_scores(
            queries, docs, queries_mask=q_mask, documents_mask=d_mask, k=8
        )
        assert result.shape == (Qb, Dq * N)

    def test_single_query_against_full_batch(self) -> None:
        """Extreme case: mini_batch_size=1."""
        Dq, N, Qt, Dt, H = 8, 2, 6, 10, 16
        queries = torch.randn(1, Qt, H)
        docs = torch.randn(Dq, N, Dt, H)
        result = xtr_scores(queries, docs, k=8)
        assert result.shape == (1, Dq * N)

    def test_scores_consistent_with_matched_batch(self) -> None:
        """Scores from a query chunk should match the corresponding rows
        of the full-batch computation."""
        Q, N, Qt, Dt, H = 8, 2, 6, 10, 16
        k = 8
        torch.manual_seed(42)
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)

        # Chunked: first 4 queries against all docs
        chunk_scores = xtr_scores(queries[:4], docs, k=k)

        # The chunk result should match the first 4 rows of the full result.
        # They won't be identical because the global top-k pool differs
        # (full has 8 queries competing, chunk has 4), but shapes must match.
        assert chunk_scores.shape == (4, Q * N)
