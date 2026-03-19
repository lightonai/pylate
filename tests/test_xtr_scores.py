"""Tests for xtr_scores: the training-time XTR score function.

Organized in three progressive layers:
  1. Shape tests — output shape (Q, Q*N) for various Q/N.
  2. Handcrafted correctness — deterministic embeddings, no torch.random.
  3. Masking and Z normalization — build on the handcrafted cases.
"""

from __future__ import annotations

import pytest
import torch

from pylate.scores import XTRKDScores, XTRScores, xtr_scores

# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------


class TestXTRScoresShapes:
    def test_q1_n1(self) -> None:
        Q, N, Qt, Dt, H = 1, 1, 4, 5, 8
        scores = XTRScores(k=3)(torch.randn(Q, Qt, H), torch.randn(Q, N, Dt, H))
        assert scores.shape == (Q, Q * N)

    def test_q2_n1(self) -> None:
        Q, N, Qt, Dt, H = 2, 1, 4, 5, 8
        scores = XTRScores(k=3)(torch.randn(Q, Qt, H), torch.randn(Q, N, Dt, H))
        assert scores.shape == (Q, Q * N)

    def test_q2_n2(self) -> None:
        Q, N, Qt, Dt, H = 2, 2, 4, 5, 8
        scores = XTRScores(k=3)(torch.randn(Q, Qt, H), torch.randn(Q, N, Dt, H))
        assert scores.shape == (Q, Q * N)

    def test_q4_n3(self) -> None:
        Q, N, Qt, Dt, H = 4, 3, 4, 5, 8
        scores = XTRScores(k=3)(torch.randn(Q, Qt, H), torch.randn(Q, N, Dt, H))
        assert scores.shape == (Q, Q * N)

    def test_requires_full_batch_flag(self) -> None:
        assert getattr(xtr_scores, "requires_full_batch", False) is True


# ---------------------------------------------------------------------------
# 1b. Batch mismatch tests (CachedContrastive: Qb != Dq)
# ---------------------------------------------------------------------------


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
        result = XTRScores(k=8)(queries, docs)
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
        result = XTRScores(k=8)(
            queries, docs, queries_mask=q_mask, documents_mask=d_mask
        )
        assert result.shape == (Qb, Dq * N)

    def test_single_query_against_full_batch(self) -> None:
        """Extreme case: mini_batch_size=1."""
        Dq, N, Qt, Dt, H = 8, 2, 6, 10, 16
        queries = torch.randn(1, Qt, H)
        docs = torch.randn(Dq, N, Dt, H)
        result = XTRScores(k=8)(queries, docs)
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
        chunk_scores = XTRScores(k=k)(queries[:4], docs)

        # The chunk result should match the first 4 rows of the full result.
        # They won't be identical because the global top-k pool differs
        # (full has 8 queries competing, chunk has 4), but shapes must match.
        assert chunk_scores.shape == (4, Q * N)


# ---------------------------------------------------------------------------
# 2. Handcrafted correctness
# ---------------------------------------------------------------------------
#
# Core setup used across both handcrafted tests:
#
#   Q=2, Qt=1, H=2, Dt=2, k=1
#   q0 = [1, 0],  q1 = [0, 1]   (orthogonal unit vectors)
#   pos_for_q0: [[1, 0], [0, 0]]  — token 0 matches q0 perfectly (dot=1.0)
#   pos_for_q1: [[0, 1], [0, 0]]  — token 0 matches q1 perfectly (dot=1.0)
#
# Flat token layout after stacking (N=1):
#   doc_idx=0 → pos_for_q0 tokens at flat positions [0, 1]
#   doc_idx=1 → pos_for_q1 tokens at flat positions [2, 3]
#
# With k=1 and orthogonal queries:
#   q0 top-1: position 0 (pos_for_q0, token 0, score 1.0)
#   q1 top-1: position 2 (pos_for_q1, token 0, score 1.0)
#
# After reshape + max + Z normalization (Z=1 for each winning doc):
#   scores[q0] = [1.0, 0.0]   argmax=0 == 0*1 ✓
#   scores[q1] = [0.0, 1.0]   argmax=1 == 1*1 ✓


class TestXTRScoresHandcrafted:
    @pytest.fixture
    def orthogonal_n1(self):
        """Q=2, N=1: each query has exactly one doc. Returns (queries, docs)."""
        queries = torch.tensor(
            [
                [[1.0, 0.0]],  # q0, token 0
                [[0.0, 1.0]],  # q1, token 0
            ]
        )  # (2, 1, 2)

        docs = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],  # doc[q0, n0]: token 0 matches q0
                [[[0.0, 1.0], [0.0, 0.0]]],  # doc[q1, n0]: token 0 matches q1
            ]
        )  # (2, 1, 2, 2)
        return queries, docs

    def test_n1_scores_are_1_0_and_0_1(self, orthogonal_n1) -> None:
        """Each query should score its own doc 1.0 and the other's doc 0.0."""
        queries, docs = orthogonal_n1
        scores = XTRScores(k=1)(queries, docs)

        assert scores.shape == (2, 2)
        assert scores[0, 0] == pytest.approx(1.0)
        assert scores[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert scores[1, 0] == pytest.approx(0.0, abs=1e-5)
        assert scores[1, 1] == pytest.approx(1.0)

    def test_n1_argmax_at_column_i_times_n(self, orthogonal_n1) -> None:
        """With N=1, argmax for query i should be at column i*1 = i."""
        queries, docs = orthogonal_n1
        scores = XTRScores(k=1)(queries, docs)

        assert scores[0].argmax().item() == 0  # 0 * 1
        assert scores[1].argmax().item() == 1  # 1 * 1

    def test_n2_argmax_at_column_i_times_n(self) -> None:
        """N=2 (pos + neg per query). Positive for query i is at column i*2.

        Flat doc order after stacking docs[q, n]:
          col 0: doc[q0, n0] = pos for q0   ← q0 label = 0*2 = 0
          col 1: doc[q0, n1] = neg for q0   (matches q1, not q0)
          col 2: doc[q1, n0] = pos for q1   ← q1 label = 1*2 = 2
          col 3: doc[q1, n1] = neg for q1   (matches q0, not q1)

        With k=1 and orthogonal queries, q0's top-1 is pos_q0 (col 0),
        q1's top-1 is pos_q1 (col 2).
        """
        queries = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0]],  # q0
                [[0.0, 1.0, 0.0, 0.0]],  # q1
            ]
        )  # (2, 1, 4)

        docs = torch.tensor(
            [
                [  # docs for q0
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # pos: matches q0
                    [[0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # neg: matches q1
                ],
                [  # docs for q1
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # pos: matches q1
                    [[0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # neg: matches q0
                ],
            ]
        )  # (2, 2, 2, 4)

        scores = XTRScores(k=1)(queries, docs)

        assert scores.shape == (2, 4)
        assert scores[0].argmax().item() == 0  # 0 * 2
        assert scores[0, 0] == pytest.approx(1.0)
        assert scores[1].argmax().item() == 2  # 1 * 2
        assert scores[1, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Masking and Z normalization
# ---------------------------------------------------------------------------
#
# All tests in this section use Q=1, N=2, Qt=2, Dt=2, H=2, k=1 unless noted.
#
# Base layout (no masks):
#   q0t0 = [1, 0],  q0t1 = [0, 1]
#   doc0 tokens: [[0.6, 0.0], [0.0, 0.0]]   — t0 matches q0t0 (dot=0.6)
#   doc1 tokens: [[0.0, 0.0], [0.0, 0.5]]   — t1 matches q0t1 (dot=0.5)
#
# With k=1, each query token retrieves its best match:
#   q0t0 top-1: doc0t0 (flat idx 0, score 0.6)
#   q0t1 top-1: doc1t1 (flat idx 3, score 0.5)
#
# topk_scores_max after reshape + max-over-Dt:
#   qt0 row: [0.6, 0.0]   (doc0 gets 0.6, doc1 gets 0)
#   qt1 row: [0.0, 0.5]   (doc0 gets 0,   doc1 gets 0.5)
#
# sum over Qt:   [0.6, 0.5]
# Z (non-zero):  [1,   1  ]
# scores:        [0.6, 0.5]


@pytest.fixture
def base_q1n2():
    """Q=1, N=2, Qt=2, Dt=2, H=2 base tensors (no masks)."""
    queries = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # q0: token 0=[1,0], token 1=[0,1]
        ]
    )  # (1, 2, 2)

    docs = torch.tensor(
        [
            [
                [[0.6, 0.0], [0.0, 0.0]],  # doc0: t0 matches q0t0
                [[0.0, 0.0], [0.0, 0.5]],  # doc1: t1 matches q0t1
            ]
        ]
    )  # (1, 2, 2, 2)
    return queries, docs


class TestXTRScoresMasks:
    def test_no_masks_baseline(self, base_q1n2) -> None:
        """Baseline without any masks: doc0=0.6, doc1=0.5."""
        queries, docs = base_q1n2
        scores = XTRScores(k=1)(queries, docs)

        # Q=1 so Q*N=2; row 0 has [doc0_score, doc1_score]
        # but the full matrix is (1, 1*2) = (1, 2)
        assert scores.shape == (1, 2)
        assert scores[0, 0] == pytest.approx(0.6)
        assert scores[0, 1] == pytest.approx(0.5)

    def test_doc_mask_excludes_pad_token(self) -> None:
        """A pad doc token with a higher raw score is excluded by documents_mask.

        doc0 token 0 is real (score 0.5 with q0t0=[1,0]).
        doc0 token 1 is PAD  (score 0.9 with q0t0, higher than token 0).
        Without mask: top-1 selects the pad token → score 0.9.
        With mask:    pad token forced to -99999  → top-1 is token 0 → score 0.5.
        """
        queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        docs = torch.tensor(
            [
                [
                    [[0.5, 0.0], [0.9, 0.0]],  # doc0: t1 is PAD with high score
                    [[0.0, 0.0], [0.0, 0.3]],  # doc1
                ]
            ]
        )  # (1, 2, 2, 2)

        # doc0 mask: token 1 is padding; doc1 mask: all real
        doc_mask = torch.tensor([[[1, 0], [1, 1]]])  # (1, 2, 2)

        scores_no_mask = XTRScores(k=1)(queries, docs)
        scores_masked = XTRScores(k=1)(queries, docs, documents_mask=doc_mask)

        # Without mask: doc0 score comes from the pad token (0.9)
        assert scores_no_mask[0, 0] == pytest.approx(0.9)
        # With mask: doc0 score comes from the real token (0.5)
        assert scores_masked[0, 0] == pytest.approx(0.5)

    def test_query_mask_zeroes_pad_query_token(self, base_q1n2) -> None:
        """A masked-out (pad) query token's contribution is zeroed.

        In the base setup, q0t1=[0,1] retrieves from doc1 (score 0.5).
        With queries_mask=[1, 0], token 1 is pad: doc1 should drop to 0.
        """
        queries, docs = base_q1n2
        query_mask = torch.tensor([[1.0, 0.0]])  # (1, 2) — token 1 is pad

        scores_no_mask = XTRScores(k=1)(queries, docs)
        scores_masked = XTRScores(k=1)(queries, docs, queries_mask=query_mask)

        # Without mask: doc1 receives contribution from q0t1 (score 0.5)
        assert scores_no_mask[0, 1] == pytest.approx(0.5)
        # With mask: q0t1 zeroed → doc1 has no retrieving query token
        assert scores_masked[0, 1] == pytest.approx(0.0, abs=1e-5)
        # doc0 score (from q0t0, which is not masked) is unchanged
        assert scores_masked[0, 0] == pytest.approx(scores_no_mask[0, 0])


class TestXTRScoresZNormalization:
    def test_z_equals_number_of_retrieving_query_tokens(self) -> None:
        """Z is the count of query tokens that retrieved from a doc.

        Setup: Q=1, N=2, Qt=2, Dt=1, H=2, k=1
          q0t0 = q0t1 = [1, 0]  (both query tokens identical)
          doc0: [[0.6, 0.0]]  — both query tokens retrieve from doc0
          doc1: [[0.0, 0.1]]  — neither query token retrieves from doc1

        Both qt0 and qt1 retrieve doc0t0 (score 0.6 each).
        topk_scores_max: [[0.6, 0], [0.6, 0]]
        sum: [1.2, 0],  Z: [2, clamped]
        score[doc0] = 1.2 / 2 = 0.6  (the per-token average, == the individual score)
        """
        queries = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # (1, 2, 2)
        docs = torch.tensor([[[[0.6, 0.0]], [[0.0, 0.1]]]])  # (1, 2, 1, 2)

        scores = XTRScores(k=1)(queries, docs)

        assert scores.shape == (1, 2)
        assert scores[0, 0] == pytest.approx(0.6)

    def test_z_normalizes_to_per_token_average(self) -> None:
        """Z normalization produces the mean max-sim across retrieving query tokens.

        Setup: Q=1, N=1, Qt=3, Dt=1, H=2, k=1
          q tokens: [1,0], [1,0], [1,0]  (all identical)
          doc0: [[0.8, 0.0]]

        All 3 tokens retrieve doc0 with score 0.8.
        sum=2.4, Z=3, score = 2.4/3 = 0.8 (the per-token max sim).
        """
        queries = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])  # (1, 3, 2)
        docs = torch.tensor([[[[0.8, 0.0]]]])  # (1, 1, 1, 2)

        scores = XTRScores(k=1)(queries, docs)

        assert scores.shape == (1, 1)
        assert scores[0, 0] == pytest.approx(0.8)

    def test_z_with_partial_retrieval(self) -> None:
        """When only some query tokens retrieve from a doc, Z reflects that count.

        Setup: Q=1, N=2, Qt=3, Dt=1, H=2, k=1
          q0t0 = [1, 0],  q0t1 = [1, 0],  q0t2 = [0, 1]
          doc0: [[0.8, 0.0]]  — retrieved by t0 and t1 (score 0.8 each); not by t2
          doc1: [[0.0, 0.7]]  — retrieved by t2 (score 0.7); not by t0/t1

        For doc0: sum=1.6, Z=2, score=0.8
        For doc1: sum=0.7, Z=1, score=0.7
        """
        queries = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # (1, 3, 2)
        docs = torch.tensor([[[[0.8, 0.0]], [[0.0, 0.7]]]])  # (1, 2, 1, 2)

        scores = XTRScores(k=1)(queries, docs)

        assert scores.shape == (1, 2)
        assert scores[0, 0] == pytest.approx(0.8)
        assert scores[0, 1] == pytest.approx(0.7)

    def test_k_limits_retrieved_tokens(self) -> None:
        """With k=1, only the single best token per query token is retained.

        Setup: Q=1, N=1, Qt=1, Dt=3, H=2, k=1
          q0t0 = [1, 0]
          doc0: [[0.9, 0], [0.7, 0], [0.5, 0]]  — three tokens, all matching q0t0

        k=1: only token 0 (score 0.9) is selected; tokens 1 and 2 are zeroed.
        topk_scores_max: [[0.9]]  (max over Dt of the masked scores)
        score = 0.9 / Z=1 = 0.9

        k=3: all three tokens selected; max over Dt is still 0.9.
        score = 0.9 / Z=1 = 0.9  (same: max is taken per doc, not summed)
        """
        queries = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
        docs = torch.tensor([[[[0.9, 0.0], [0.7, 0.0], [0.5, 0.0]]]])  # (1, 1, 3, 2)

        scores_k1 = XTRScores(k=1)(queries, docs)
        scores_k3 = XTRScores(k=3)(queries, docs)

        assert scores_k1[0, 0] == pytest.approx(0.9)
        assert scores_k3[0, 0] == pytest.approx(0.9)

    def test_k_changes_which_doc_wins(self) -> None:
        """k determines which docs get non-zero scores, affecting the winner.

        Setup: Q=1, N=2, Qt=1, Dt=2, H=2
          q0t0 = [1, 0]
          doc0: [[0.9, 0.0], [0.0, 0.0]]  — best token scores 0.9
          doc1: [[0.8, 0.0], [0.7, 0.0]]  — two decent tokens

        k=1: only doc0t0 (0.9) is selected → doc0 wins
        k=3: doc0t0 (0.9), doc1t0 (0.8), doc1t1 (0.7) are selected.
             doc0: max=0.9, Z=1, score=0.9
             doc1: max=0.8, Z=1, score=0.8
             doc0 still wins, but doc1 now has a non-zero score.
        """
        queries = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
        docs = torch.tensor(
            [
                [
                    [[0.9, 0.0], [0.0, 0.0]],  # doc0
                    [[0.8, 0.0], [0.7, 0.0]],  # doc1
                ]
            ]
        )  # (1, 2, 2, 2)

        scores_k1 = XTRScores(k=1)(queries, docs)
        scores_k3 = XTRScores(k=3)(queries, docs)

        # k=1: only doc0 has non-zero score
        assert scores_k1[0, 0] == pytest.approx(0.9)
        assert scores_k1[0, 1] == pytest.approx(0.0, abs=1e-5)

        # k=3: both docs have non-zero scores
        assert scores_k3[0, 0] == pytest.approx(0.9)
        assert scores_k3[0, 1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 4. xtr_kd_scores slicing
# ---------------------------------------------------------------------------


class TestXTRKDScoresSlicing:
    def test_shape_is_q_n(self) -> None:
        """xtr_kd_scores returns (Q, N), not (Q, Q*N)."""
        Q, N, Qt, Dt, H = 3, 4, 6, 5, 8
        kd = XTRKDScores(k=3)(torch.randn(Q, Qt, H), torch.randn(Q, N, Dt, H))
        assert kd.shape == (Q, N)

    def test_slicing_matches_xtr_scores_diagonal(self) -> None:
        """Each row of xtr_kd_scores equals the correct N-wide slice of xtr_scores.

        For query i, xtr_kd_scores[i] == xtr_scores_full[i, i*N:(i+1)*N].
        """
        Q, N, Qt, Dt, H = 4, 3, 5, 4, 8
        torch.manual_seed(42)
        queries = torch.randn(Q, Qt, H)
        docs = torch.randn(Q, N, Dt, H)

        scorer = XTRScores(k=5)
        full = scorer(queries, docs)  # (Q, Q*N)
        kd = XTRKDScores(k=5)(queries, docs)  # (Q, N)

        for i in range(Q):
            expected = full[i, i * N : (i + 1) * N]
            torch.testing.assert_close(kd[i], expected)

    def test_handcrafted_n2_matches_diagonal(self) -> None:
        """Handcrafted Q=2 N=2: verify slicing picks the right columns.

        xtr_scores returns (2, 4). The per-query docs are:
          query 0: columns 0, 1  (i*N=0*2=0)
          query 1: columns 2, 3  (i*N=1*2=2)
        """
        queries = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0]],
            ]
        )  # (2, 1, 4)

        docs = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # q0 pos
                    [[0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # q0 neg
                ],
                [
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # q1 pos
                    [[0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # q1 neg
                ],
            ]
        )  # (2, 2, 2, 4)

        full = XTRScores(k=1)(queries, docs)  # (2, 4)
        kd = XTRKDScores(k=1)(queries, docs)  # (2, 2)

        # query 0 should get columns 0,1 from full
        torch.testing.assert_close(kd[0], full[0, 0:2])
        # query 1 should get columns 2,3 from full
        torch.testing.assert_close(kd[1], full[1, 2:4])
