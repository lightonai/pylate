from __future__ import annotations

import numpy as np
import pytest
import torch

from pylate import indexes

pytest.importorskip("scann")


def test_scann_is_exported() -> None:
    """ScaNN should be importable from pylate.indexes."""
    assert hasattr(indexes, "ScaNN")


@pytest.mark.parametrize(
    ("verbose_input", "expected_level"),
    [
        (False, "none"),
        (True, "init"),
        ("none", "none"),
        ("init", "init"),
        ("all", "all"),
    ],
)
def test_scann_verbose_normalization(verbose_input: bool | str, expected_level: str) -> None:
    """ScaNN should normalize bool/string verbosity to internal levels."""
    index = indexes.ScaNN(verbose=verbose_input)
    assert index.verbose_level == expected_level
    assert index.verbose == (expected_level in ("init", "all"))


def test_scann_verbose_level_alias_overrides_verbose() -> None:
    """Backward-compatible verbose_level should override verbose."""
    index = indexes.ScaNN(verbose=False, verbose_level="all")
    assert index.verbose_level == "all"
    assert index.verbose is True


def test_scann_invalid_verbose_level_raises() -> None:
    """Unsupported verbose values should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid verbosity level"):
        indexes.ScaNN(verbose="loud")


def test_scann_add_documents_returns_self_and_preserves_fp16_storage() -> None:
    """Stored flattened embeddings should preserve incoming fp16 dtype."""
    index = indexes.ScaNN(store_embeddings=True)

    documents_ids = ["d1", "d2"]
    documents_embeddings = [
        torch.randn(12, 8, dtype=torch.float16),
        torch.randn(10, 8, dtype=torch.float16),
    ]

    returned = index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
        batch_size=2,
    )

    assert returned is index
    assert index.flattened_embeddings is not None
    assert index.flattened_embeddings.dtype == np.float16
    assert index.flattened_embeddings.shape == (22, 8)


@pytest.mark.parametrize("docs_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("queries_dtype", [torch.float16, torch.float32])
def test_scann_accepts_fp16_and_fp32_documents_and_queries(
    docs_dtype: torch.dtype, queries_dtype: torch.dtype
) -> None:
    """ScaNN should accept both fp16/fp32 docs and fp16/fp32 queries."""
    index = indexes.ScaNN(store_embeddings=True)
    index.add_documents(
        documents_ids=["d1", "d2", "d3"],
        documents_embeddings=[
            torch.randn(8, 8, dtype=docs_dtype),
            torch.randn(7, 8, dtype=docs_dtype),
            torch.randn(9, 8, dtype=docs_dtype),
        ],
        batch_size=3,
    )

    expected_np_dtype = np.float16 if docs_dtype == torch.float16 else np.float32
    assert index.flattened_embeddings is not None
    assert index.flattened_embeddings.dtype == expected_np_dtype

    results = index(
        queries_embeddings=[torch.randn(2, 8, dtype=queries_dtype)],
        k=2,
    )

    assert set(results.keys()) == {"documents_ids", "distances"}
    assert len(results["documents_ids"]) == 1
    assert len(results["documents_ids"][0]) == 2
    assert len(results["documents_ids"][0][0]) == 2


def test_scann_errors_on_mixed_embedding_dtypes() -> None:
    """All document tensors must share dtype (fp16 or fp32)."""
    index = indexes.ScaNN(store_embeddings=False)

    with pytest.raises(ValueError, match="same dtype"):
        index.add_documents(
            documents_ids=["d1", "d2"],
            documents_embeddings=[
                torch.randn(2, 8, dtype=torch.float16),
                torch.randn(2, 8, dtype=torch.float32),
            ],
            batch_size=2,
        )


@pytest.mark.parametrize("docs_dtype", [torch.float16, torch.float32])
def test_scann_get_documents_embeddings_by_docid(docs_dtype: torch.dtype) -> None:
    """Stored embeddings should be retrievable by document ID."""
    index = indexes.ScaNN(store_embeddings=True)

    d1 = torch.arange(0, 96, dtype=torch.float32).reshape(12, 8).to(docs_dtype)
    d2 = torch.arange(96, 176, dtype=torch.float32).reshape(10, 8).to(docs_dtype)
    index.add_documents(
        documents_ids=["d1", "d2"],
        documents_embeddings=[d1, d2],
        batch_size=2,
    )

    retrieved = index.get_documents_embeddings([["d2", "d1"]])
    assert len(retrieved) == 1
    assert len(retrieved[0]) == 2
    assert retrieved[0][0].dtype == (np.float16 if docs_dtype == torch.float16 else np.float32)
    assert np.array_equal(retrieved[0][0], d2.cpu().numpy())
    assert np.array_equal(retrieved[0][1], d1.cpu().numpy())


def test_scann_get_documents_embeddings_requires_store_embeddings() -> None:
    """Accessing embeddings without store_embeddings should raise clearly."""
    index = indexes.ScaNN(store_embeddings=False)

    with pytest.raises(NotImplementedError, match="store_embeddings=True"):
        index.get_documents_embeddings([["d1"]])


def test_scann_get_documents_embeddings_missing_docid_raises() -> None:
    """Unknown document IDs should raise a ValueError."""
    index = indexes.ScaNN(store_embeddings=True)
    index.add_documents(
        documents_ids=["d1", "d2"],
        documents_embeddings=[
            torch.randn(12, 8, dtype=torch.float32),
            torch.randn(10, 8, dtype=torch.float32),
        ],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="not found in index"):
        index.get_documents_embeddings([["d3"]])
