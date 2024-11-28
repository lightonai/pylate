import math

import torch

from pylate import models, rank


def test_model_creation(**kwargs) -> None:
    """Test the creation of different models."""
    query = ["fruits are healthy."]
    documents = [["fruits are healthy.", "fruits are good for health."]]
    torch.manual_seed(42)
    # Creation from a base encoder
    model = models.ColBERT(model_name_or_path="bert-base-uncased")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 25.92, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 23.7, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from a base sentence-transformer
    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 18.77, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 18.63, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from stanford-nlp (safetensor)
    model = models.ColBERT(model_name_or_path="answerdotai/answerai-colbert-small-v1")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 31.71, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 31.64, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from stanford-nlp (bin)
    model = models.ColBERT(model_name_or_path="Crystalcareai/Colbertv2")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 31.15, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 30.61, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from PyLate
    model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 30.01, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 26.98, rel_tol=0.01, abs_tol=0.01
    )
