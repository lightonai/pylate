from __future__ import annotations

import numpy as np
import pytest
import torch

from pylate import indexes

# mlxvs runs on the GPU of Apple silicon Macs only
pytest.importorskip("mlxvs")


def _unit(n: int, dim: int = 16, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))
    return torch.nn.functional.normalize(x, dim=-1).to(dtype)


def _documents(n_docs: int, dim: int = 16, seed: int = 0, dtype=torch.float32):
    rng = np.random.default_rng(seed)
    ids = [f"doc{i}" for i in range(n_docs)]
    embeddings = [
        _unit(int(rng.integers(3, 12)), dim, seed + i, dtype) for i in range(n_docs)
    ]
    return ids, embeddings


def _exact_neighbors(ids, embeddings, query, k):
    """The document of each of the k tokens of largest inner product with each query token."""
    tokens = torch.cat(embeddings).float()
    owners = np.array(
        [doc_id for doc_id, e in zip(ids, embeddings) for _ in range(len(e))]
    )
    order = torch.argsort(-(query.float() @ tokens.T), dim=1)[:, :k].numpy()
    return owners[order]


def test_mlxvs_is_exported() -> None:
    assert hasattr(indexes, "MLXVS")


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mlxvs_matches_exact_token_search(exact: bool, dtype: torch.dtype) -> None:
    """Every query token's neighbors are its documents' tokens of largest inner product."""
    ids, embeddings = _documents(300, dtype=dtype)
    index = indexes.MLXVS(embedding_size=16, exact=exact)
    assert index.add_documents(ids, embeddings) is index

    query = _unit(6, seed=999)
    results = index(queries_embeddings=[query], k=5)
    assert set(results.keys()) == {"documents_ids", "distances"}
    assert len(results["documents_ids"]) == 1 and len(results["documents_ids"][0]) == 6

    expected = _exact_neighbors(ids, embeddings, query, 5)
    got = np.array(results["documents_ids"][0])
    agreement = np.mean(
        [len(set(g) & set(e)) / len(set(e)) for g, e in zip(got, expected)]
    )
    assert agreement >= (1.0 if exact else 0.9)
    # cosine distances, nearest first
    distances = np.array(results["distances"][0])
    assert np.all(np.diff(distances, axis=1) >= -1e-3)
    assert distances.min() >= -1e-3 and distances.max() <= 2 + 1e-3


@pytest.mark.parametrize("exact", [True, False])
def test_mlxvs_add_remove_and_readd(exact: bool) -> None:
    ids, embeddings = _documents(200)
    index = indexes.MLXVS(embedding_size=16, exact=exact)
    index.add_documents(ids[:120], embeddings[:120])
    index.add_documents(ids[120:], embeddings[120:])

    # a query equal to a document's first token finds that document first
    results = index(queries_embeddings=[embeddings[150][:1]], k=3)
    assert results["documents_ids"][0][0][0] == "doc150"

    index.remove_documents(["doc150", "unknown"])
    results = index(queries_embeddings=[embeddings[150][:1]], k=50)
    assert "doc150" not in set(results["documents_ids"][0][0])
    with pytest.raises(ValueError, match="not found"):
        index.get_documents_embeddings([["doc150"]])

    index.add_documents(["doc150"], [embeddings[150]])
    results = index(queries_embeddings=[embeddings[150][:1]], k=3)
    assert results["documents_ids"][0][0][0] == "doc150"
    np.testing.assert_allclose(
        index.get_documents_embeddings([["doc150"]])[0][0], embeddings[150].numpy()
    )


def test_mlxvs_get_documents_embeddings_preserves_dtype() -> None:
    ids, embeddings = _documents(10, dtype=torch.float16)
    index = indexes.MLXVS(embedding_size=16)
    index.add_documents(ids, embeddings)
    got = index.get_documents_embeddings([["doc3", "doc7"], ["doc0"]])
    assert [len(group) for group in got] == [2, 1]
    assert got[0][0].dtype == np.float16
    np.testing.assert_array_equal(got[0][1], embeddings[7].numpy())


def test_mlxvs_accepts_numpy_and_rejects_mixed_dtypes() -> None:
    index = indexes.MLXVS(embedding_size=8)
    index.add_documents(["a"], [np.random.rand(4, 8).astype(np.float32)])
    with pytest.raises(ValueError, match="same dtype"):
        index.add_documents(["b"], [np.random.rand(4, 8).astype(np.float16)])
    with pytest.raises(ValueError, match="float16 or float32"):
        index.add_documents(["c"], [np.random.rand(4, 8).astype(np.float64)])


def test_mlxvs_empty_index_raises() -> None:
    with pytest.raises(ValueError, match="Index is empty"):
        indexes.MLXVS(embedding_size=8)(queries_embeddings=[_unit(2, 8)], k=2)


@pytest.mark.parametrize("exact", [True, False])
def test_mlxvs_persists_to_index_folder(tmp_path, exact: bool) -> None:
    ids, embeddings = _documents(150)
    index = indexes.MLXVS(
        embedding_size=16, exact=exact, index_folder=str(tmp_path), index_name="test"
    )
    index.add_documents(ids, embeddings)
    index.remove_documents(["doc3"])
    query = [_unit(4, seed=5)]
    expected = index(queries_embeddings=query, k=4)

    loaded = indexes.MLXVS(
        embedding_size=16, exact=exact, index_folder=str(tmp_path), index_name="test"
    )
    np.testing.assert_array_equal(
        np.array(loaded(queries_embeddings=query, k=4)["documents_ids"]),
        np.array(expected["documents_ids"]),
    )
    with pytest.raises(ValueError, match="not found"):
        loaded.get_documents_embeddings([["doc3"]])

    fresh = indexes.MLXVS(
        embedding_size=16,
        exact=exact,
        index_folder=str(tmp_path),
        index_name="test",
        override=True,
    )
    assert fresh.embeddings is None


def test_mlxvs_with_colbert_retriever() -> None:
    """The index plugs into the ColBERT retriever: token search, then MaxSim reranking."""
    from pylate import retrieve

    ids, embeddings = _documents(100)
    index = indexes.MLXVS(embedding_size=16)
    index.add_documents(ids, embeddings)
    retriever = retrieve.ColBERT(index=index)
    results = retriever.retrieve(queries_embeddings=[embeddings[42]], k=3)
    assert results[0][0]["id"] == "doc42"
