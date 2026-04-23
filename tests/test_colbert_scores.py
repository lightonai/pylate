"""Tests for ColBERTScores: the contrastive-training ColBERT score class.

The class takes (Q, Qt, H) queries and (Q, N, Dt, H) stacked documents and
returns (Q, Q*N) with query-major ordering — scores[i, j*N + k] is query i
vs query j's k-th document.
"""

from __future__ import annotations

import torch

from pylate.scores import ColBERTScores, colbert_scores


class TestColBERTScoresShapes:
    def test_shape_is_q_by_q_times_n(self) -> None:
        Q, N, Qt, Dt, H = 4, 3, 5, 7, 8
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)
        scores = ColBERTScores()(queries, docs)
        assert scores.shape == (Q, Q * N)


class TestColBERTScoresQueryMajorOrdering:
    def test_positive_at_column_i_times_n(self) -> None:
        """Each query's positive must score strictly higher than any other
        in-batch doc, so argmax sits at column i*N.
        """
        # Q=2, N=2, Qt=1, Dt=1, H=2. Positives score 1.0, negatives 0.5.
        queries = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # (2, 1, 2)
        docs = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 0.5]]],  # q0: pos matches q0, neg partially matches q1
                [[[0.0, 1.0]], [[0.5, 0.0]]],  # q1: pos matches q1, neg partially matches q0
            ]
        )  # (2, 2, 1, 2)

        scores = ColBERTScores()(queries, docs)

        assert scores.shape == (2, 4)
        # Query-major columns: [q0_pos, q0_neg, q1_pos, q1_neg]
        assert scores[0].argmax().item() == 0  # q0 positive at i*N = 0
        assert scores[1].argmax().item() == 2  # q1 positive at i*N = 2


class TestColBERTScoresEquivalenceWithLegacy:
    """The new class must produce the same scores as the legacy per-group loop
    over colbert_scores (modulo column reordering: group-major vs query-major)."""

    def test_matches_per_group_loop(self) -> None:
        Q, N, Qt, Dt, H = 4, 3, 5, 7, 8
        torch.manual_seed(0)
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)

        new = ColBERTScores()(queries, docs)  # (Q, Q*N), query-major

        # Legacy: loop over N groups, each call gives (Q, Q); concat along dim=1
        # producing group-major layout (Q, Q*N) where block j spans cols [j*Q, (j+1)*Q).
        legacy_group_major = torch.cat(
            [colbert_scores(queries, docs[:, j]) for j in range(N)],
            dim=1,
        )

        # Reorder legacy to query-major: col (q*N + n) in new == col (n*Q + q) in legacy
        q_idx = torch.arange(Q).unsqueeze(1).expand(Q, N)
        n_idx = torch.arange(N).unsqueeze(0).expand(Q, N)
        perm = (n_idx * Q + q_idx).reshape(-1)
        legacy_reordered = legacy_group_major[:, perm]

        torch.testing.assert_close(new, legacy_reordered)


class TestColBERTScoresQuerySlicing:
    """Scoring a slice of queries must match the same rows of the full call —
    backs the loss-level query-chunk loop."""

    def test_slicing_matches_full_first_rows(self) -> None:
        Q, N, Qt, Dt, H = 6, 2, 5, 7, 8
        torch.manual_seed(0)
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)

        scorer = ColBERTScores()
        full = scorer(queries, docs)
        chunk = scorer(queries[:3], docs)

        assert chunk.shape == (3, Q * N)
        torch.testing.assert_close(chunk, full[:3])
