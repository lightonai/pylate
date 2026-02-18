from __future__ import annotations

import pytest
import torch

from pylate.rank import score_xtr


class TestScoreXTR:
    """Comprehensive unit tests for score_xtr()."""

    def test_basic_scoring_no_overlap(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]

        results = score_xtr(query_doc_ids, query_scores, k=6)
        assert len(results) == 6
        assert results[0]["score"] == pytest.approx(1.3, abs=1e-5)
        assert results[0]["id"] in ["doc1", "doc4"]

    def test_overlapping_documents(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc3", "doc4"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]

        results = score_xtr(query_doc_ids, query_scores, k=4)
        assert len(results) == 4
        assert results[0]["id"] == "doc2"
        assert results[0]["score"] == pytest.approx(1.5, abs=1e-5)
        assert results[1]["id"] == "doc1"
        assert results[1]["score"] == pytest.approx(1.3, abs=1e-5)

    def test_duplicate_doc_in_same_query_token_uses_max(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc1"],
            ["doc2", "doc3"],
        ]
        query_scores = [
            [0.9, 0.7, 0.6],
            [0.8, 0.5],
        ]

        results = score_xtr(query_doc_ids, query_scores, k=3)
        doc1 = next(r for r in results if r["id"] == "doc1")
        assert doc1["score"] == pytest.approx(1.4, abs=1e-5)

    def test_integer_doc_ids(self) -> None:
        query_doc_ids = [[1, 2, 3], [2, 3, 4]]
        query_scores = [[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]]

        results = score_xtr(query_doc_ids, query_scores, k=4)
        assert len(results) == 4
        assert isinstance(results[0]["id"], int)
        assert results[0]["id"] == 2
        assert results[0]["score"] == pytest.approx(1.5, abs=1e-5)

    def test_single_query_token(self) -> None:
        query_doc_ids = [["doc1", "doc2", "doc3"]]
        query_scores = [[0.9, 0.7, 0.5]]
        results = score_xtr(query_doc_ids, query_scores, k=3)
        assert len(results) == 3
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == pytest.approx(0.9, abs=1e-5)

    def test_k_larger_than_unique_docs(self) -> None:
        query_doc_ids = [["doc1", "doc2"], ["doc2", "doc3"]]
        query_scores = [[0.9, 0.7], [0.8, 0.5]]
        results = score_xtr(query_doc_ids, query_scores, k=10)
        assert len(results) == 3

    def test_empty_query(self) -> None:
        assert score_xtr([], [], k=5) == []

    def test_cuda_device(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        query_doc_ids = [["doc1", "doc2"], ["doc2", "doc3"]]
        query_scores = [[0.9, 0.7], [0.8, 0.5]]
        results = score_xtr(query_doc_ids, query_scores, k=3, device="cuda")
        assert len(results) == 3
        assert results[0]["id"] == "doc2"

    def test_score_imputation_min(self) -> None:
        query_doc_ids = [["doc1"], ["doc2"]]
        query_scores = [[0.9], [0.5]]
        results = score_xtr(query_doc_ids, query_scores, k=2, imputation="min")
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(1.4, abs=1e-5)
        assert results[1]["score"] == pytest.approx(1.4, abs=1e-5)

    def test_imputation_modes_execute(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc3", "doc4"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]
        for imputation in ["zero", "mean", "percentile", "power_law"]:
            results = score_xtr(
                query_doc_ids=query_doc_ids,
                query_scores=query_scores,
                k=4,
                imputation=imputation,
                percentile=10.0,
                power_law_multiplier=100.0,
            )
            assert len(results) == 4

    def test_many_query_tokens(self) -> None:
        n_tokens = 10
        query_doc_ids = [
            [f"doc{i}", f"doc{i + 1}", f"doc{i + 2}"] for i in range(n_tokens)
        ]
        query_scores = [[0.9, 0.7, 0.5] for _ in range(n_tokens)]
        results = score_xtr(query_doc_ids, query_scores, k=5)
        assert len(results) == 5
        for result in results:
            assert "id" in result
            assert "score" in result
            assert result["score"] > 0

    def test_result_structure(self) -> None:
        query_doc_ids = [["doc1", "doc2"]]
        query_scores = [[0.9, 0.7]]
        results = score_xtr(query_doc_ids, query_scores, k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_descending_order(self) -> None:
        query_doc_ids = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        query_scores = [[0.3, 0.9, 0.5, 0.1, 0.7]]
        results = score_xtr(query_doc_ids, query_scores, k=5)
        assert [r["id"] for r in results] == ["doc2", "doc5", "doc3", "doc1", "doc4"]
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


class TestScoreXTREdgeCases:
    def test_negative_scores(self) -> None:
        query_doc_ids = [["doc1", "doc2"], ["doc2", "doc3"]]
        query_scores = [[0.5, -0.3], [0.2, -0.5]]
        results = score_xtr(query_doc_ids, query_scores, k=3)
        assert len(results) == 3

    def test_all_same_scores(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        query_scores = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
        results = score_xtr(query_doc_ids, query_scores, k=6)
        assert len(results) == 6
        for result in results:
            assert result["score"] == pytest.approx(1.0, abs=1e-5)

    def test_very_small_k(self) -> None:
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc3", "doc4"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]
        results = score_xtr(query_doc_ids, query_scores, k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc2"

    def test_raises_on_mismatched_query_lengths(self) -> None:
        with pytest.raises(ValueError, match="same number of query tokens"):
            score_xtr(
                query_doc_ids=[["doc1"], ["doc2"]],
                query_scores=[[0.9]],
                k=2,
            )

    def test_raises_on_token_length_mismatch(self) -> None:
        with pytest.raises(
            ValueError, match="matching document IDs and scores lengths"
        ):
            score_xtr(
                query_doc_ids=[["doc1", "doc2"]],
                query_scores=[[0.9]],
                k=2,
            )
