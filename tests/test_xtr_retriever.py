from __future__ import annotations

import pytest
import torch

from pylate import indexes, retrieve

pytest.importorskip("scann")


def _build_tiny_scann_index() -> indexes.ScaNN:
    index = indexes.ScaNN(
        index_name="xtr_test_index",
        store_embeddings=False,
        override=True,
        verbose=False,
    )
    # Keep enough vectors for ScaNN AH training on tiny tests.
    docs = [
        torch.randn(10, 8, dtype=torch.float32),
        torch.randn(8, 8, dtype=torch.float32),
        torch.randn(9, 8, dtype=torch.float32),
    ]
    index.add_documents(
        documents_ids=["d1", "d2", "d3"],
        documents_embeddings=docs,
        batch_size=3,
    )
    return index


def test_xtr_is_exported() -> None:
    assert hasattr(retrieve, "XTR")


def test_xtr_constructor_rejects_plaid_style_indices() -> None:
    # Use __new__ to avoid expensive backend initialization while still testing
    # the isinstance check against PLAID-style indices.
    plaid_like = indexes.PLAID.__new__(indexes.PLAID)
    with pytest.raises(ValueError, match="non end-to-end index"):
        retrieve.XTR(index=plaid_like)


def test_xtr_retrieve_subset_not_supported() -> None:
    retriever = retrieve.XTR(index=_build_tiny_scann_index())
    with pytest.raises(
        NotImplementedError, match="Subset filtering is not implemented"
    ):
        retriever.retrieve(
            queries_embeddings=[torch.randn(2, 8, dtype=torch.float32)],
            k=2,
            k_token=5,
            subset=["d1"],
        )


def test_xtr_retrieve_e2e_with_scann() -> None:
    retriever = retrieve.XTR(index=_build_tiny_scann_index(), verbose=False)
    queries = [
        torch.randn(3, 8, dtype=torch.float32),
        torch.randn(2, 8, dtype=torch.float32),
    ]

    results = retriever.retrieve(
        queries_embeddings=queries,
        k=2,
        k_token=5,
    )

    # E2E shape check: one result list per query.
    assert len(results) == 2

    # Content check: each query returns at most k documents.
    assert all(len(r) <= 2 for r in results)

    # Schema check: every item is a dict with id/score fields.
    for query_results in results:
        for item in query_results:
            assert isinstance(item, dict)
            assert "id" in item
            assert "score" in item


def test_xtr_retrieve_single_query_2d_input() -> None:
    """A single query passed as a 2D tensor (num_tokens, dim) should be
    treated as one query, not num_tokens queries."""
    retriever = retrieve.XTR(index=_build_tiny_scann_index(), verbose=False)
    single_query = torch.randn(4, 8, dtype=torch.float32)  # 2D, not wrapped in list

    results = retriever.retrieve(
        queries_embeddings=single_query,
        k=2,
        k_token=5,
    )

    assert len(results) == 1
    assert all(len(r) <= 2 for r in results)
