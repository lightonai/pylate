from __future__ import annotations

from pylate import indexes, models, retrieve


def test_voyager_index(**kwargs) -> None:
    """Test the Voyager index class."""

    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

    configurations = [
        {
            "sentences": ["fruits are healthy.", "fruits are good for health."],
            "convert_to_tensor": True,
            "batch_size": 1,
            "document_ids": ["1", "2"],
        },
        {
            "sentences": ["fruits are healthy.", "fruits are good for health."],
            "convert_to_tensor": False,
            "batch_size": 1,
            "document_ids": ["1", "2"],
        },
        {
            "sentences": "fruits are healthy.",
            "convert_to_tensor": True,
            "batch_size": 1,
            "document_ids": ["1"],
        },
        {
            "sentences": "fruits are healthy.",
            "convert_to_tensor": False,
            "batch_size": 1,
            "document_ids": ["1"],
        },
    ]

    for configuration in configurations:
        documents_ids = configuration.pop("document_ids")

        index = indexes.Voyager(
            index_folder="test_indexes",
            index_name="colbert",
            override=True,
            embedding_size=128,
        )

        documents_embeddings = model.encode(
            **configuration,
            is_query=False,
        )

        index.add_documents(
            documents_ids=documents_ids, documents_embeddings=documents_embeddings
        )

        queries_embeddings = model.encode(
            **configuration,
            is_query=True,
        )

        matchs = index(queries_embeddings, k=5)

        assert isinstance(matchs, dict)
        assert "documents_ids" in matchs
        assert "distances" in matchs

        assert (
            matchs["distances"].shape[0] == len(configuration["sentences"])
            if isinstance(configuration["sentences"], list)
            else 1
        )

        retriever = retrieve.ColBERT(index=index)
        results = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=5,
        )

        assert isinstance(results, list)
        assert (
            len(results) == len(configuration["sentences"])
            if isinstance(configuration["sentences"], list)
            else 1
        )

        for query_results in results:
            for result in query_results:
                assert "id" in result
                assert "score" in result
