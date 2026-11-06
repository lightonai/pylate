"""Regression tests: padded document tokens must be excluded from MaxSim.

Before the fix, ``documents_mask`` was applied by multiplication, which
floored every query-token max at 0: whenever a query token's real
similarities to a document were all negative, the padded (zero) slot won
the max instead of the true maximum. Padding then changed the score and
could even reorder a ranking.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylate.rank import rerank
from pylate.scores import colbert_kd_scores, colbert_scores


class TestMaskedPaddingExclusion:
    def test_padded_score_equals_dense_score(self) -> None:
        """Adding masked padding tokens must not change the score."""
        q = torch.tensor([[[-1.0], [-1.0]]])
        doc_real = torch.tensor([[[1.0], [1.0]]])
        doc_padded = torch.tensor([[[1.0], [1.0], [0.0]]])
        doc_mask = torch.tensor([[1.0, 1.0, 0.0]])

        dense = colbert_scores(q, doc_real)
        padded = colbert_scores(q, doc_padded, documents_mask=doc_mask)

        torch.testing.assert_close(padded, dense)

    def test_no_mask_call_unchanged(self) -> None:
        q = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.5]]])
        torch.testing.assert_close(colbert_scores(q, d), torch.tensor([[3.0, 3.0]]))

    def test_kd_padded_score_equals_dense(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        docs_real = torch.tensor([[[[1.0], [1.0]], [[1.0], [1.0]]]])
        docs_padded = torch.tensor([[[[1.0], [1.0], [0.0]], [[1.0], [1.0], [0.0]]]])
        doc_mask = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])

        dense = colbert_kd_scores(q, docs_real)
        padded = colbert_kd_scores(q, docs_padded, documents_mask=doc_mask)

        torch.testing.assert_close(padded, dense)

    @pytest.mark.parametrize(
        "device",
        [
            None,
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ),
        ],
    )
    def test_rerank_unequal_lengths_are_not_reordered_by_padding(self, device) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        # B has worse real similarities, but its zero padding used to win the max.
        results = rerank(
            documents_ids=[["A", "B"]],
            queries_embeddings=q,
            documents_embeddings=[
                [torch.tensor([[1.0], [1.0]]), torch.tensor([[2.0]])],
            ],
            device=device,
        )
        assert results == [[{"id": "A", "score": -2.0}, {"id": "B", "score": -4.0}]]

    @pytest.mark.parametrize("array", [torch.tensor, np.array])
    def test_rerank_accepts_integer_embeddings(self, array) -> None:
        results = rerank(
            documents_ids=[["A", "B"]],
            queries_embeddings=[array([[-1], [-1]])],
            documents_embeddings=[[array([[1], [1]]), array([[2]])]],
        )

        assert results == [[{"id": "A", "score": -2.0}, {"id": "B", "score": -4.0}]]

    def test_padded_query_token_contributes_zero_no_nan(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        q_mask = torch.tensor([[1.0, 0.0]])
        doc = torch.tensor([[[1.0], [0.0]]])
        doc_mask = torch.tensor([[1.0, 0.0]])

        out = colbert_scores(q, doc, queries_mask=q_mask, documents_mask=doc_mask)

        assert not torch.isnan(out).any()
        torch.testing.assert_close(out, torch.tensor([[-1.0]]))

    def test_fully_invalid_doc_row_is_inf_not_nan(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        doc = torch.tensor([[[1.0], [1.0]]])
        doc_mask = torch.tensor([[0.0, 0.0]])

        out = colbert_scores(q, doc, documents_mask=doc_mask)

        assert not torch.isnan(out).any()
        assert torch.isneginf(out).all()

    @pytest.mark.parametrize("scorer", [colbert_scores, colbert_kd_scores])
    def test_masked_tokens_do_not_receive_gradients(self, scorer) -> None:
        q = torch.tensor([[[-1.0], [-2.0]]], requires_grad=True)
        doc = torch.tensor([[[1.0], [2.0], [0.0]]], requires_grad=True)
        doc_mask = torch.tensor([[1.0, 1.0, 0.0]])
        kd = scorer is colbert_kd_scores

        out = scorer(
            q,
            doc.unsqueeze(0) if kd else doc,
            queries_mask=torch.tensor([[1.0, 0.0]]),
            documents_mask=doc_mask.unsqueeze(0) if kd else doc_mask,
            backend="torch",
        )
        out.sum().backward()

        torch.testing.assert_close(out, torch.tensor([[-1.0]]))
        torch.testing.assert_close(q.grad, torch.tensor([[[1.0], [0.0]]]))
        torch.testing.assert_close(doc.grad, torch.tensor([[[-1.0], [0.0], [0.0]]]))

    @pytest.mark.parametrize("scorer", [colbert_scores, colbert_kd_scores])
    def test_fully_masked_queries_and_documents_score_zero(self, scorer) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        doc = torch.tensor([[[1.0], [1.0]]])
        doc_mask = torch.zeros(1, 2)
        kd = scorer is colbert_kd_scores

        out = scorer(
            q,
            doc.unsqueeze(0) if kd else doc,
            queries_mask=torch.zeros(1, 2),
            documents_mask=doc_mask.unsqueeze(0) if kd else doc_mask,
            backend="torch",
        )

        torch.testing.assert_close(out, torch.zeros(1, 1))
