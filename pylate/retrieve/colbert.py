from __future__ import annotations

import numpy as np
import torch

from ..rank import RerankResult, rerank
from .base import BaseRetriever


class ColBERT(BaseRetriever):
    """ColBERT retriever.

    Parameters
    ----------
    index
        The index to use for retrieval. Any index that extends ``Base``
        (e.g. PLAID, Voyager, ScaNN).

    Examples
    --------
    >>> from pylate import indexes, models, retrieve

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     device="cpu",
    ... )

    >>> documents_ids = [f"document_id_{i}" for i in range(20)]
    >>> documents = [f"This is the content of document {i}." for i in range(20)]

    >>> documents_embeddings = model.encode(
    ...     sentences=documents,
    ...     batch_size=1,
    ...     is_query=False,
    ... )

    >>> index = indexes.PLAID(
    ...     index_folder="test_indexes",
    ...     index_name="colbert",
    ...     override=True,
    ... )

    >>> index = index.add_documents(
    ...     documents_ids=documents_ids,
    ...     documents_embeddings=documents_embeddings,
    ... )
    Computing centroids of embeddings.
    Creating FastPlaid index.

    >>> retriever = retrieve.ColBERT(index=index)

    >>> queries_embeddings = model.encode(
    ...     ["fruits are healthy.", "fruits are good for health."],
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 2

    >>> queries_embeddings = model.encode(
    ...     "fruits are healthy.",
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 1

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ...     subset=["document_id_10"],
    ... )

    """

    default_k_token = 100
    default_batch_size = 50
    progress_desc = "Retrieving documents"

    def _score_batch(
        self,
        batch_queries_embeddings: list | np.ndarray | torch.Tensor,
        hits: dict,
        *,
        k: int,
        device: str,
    ) -> list[list[RerankResult]]:
        documents_ids = [
            list(
                {
                    document_id
                    for query_token_document_ids in query_documents_ids
                    for document_id in query_token_document_ids
                }
            )
            for query_documents_ids in hits["documents_ids"]
        ]
        documents_embeddings = self.index.get_documents_embeddings(documents_ids)
        reranked = rerank(
            documents_ids=documents_ids,
            queries_embeddings=batch_queries_embeddings,
            documents_embeddings=documents_embeddings,
            device=device,
        )
        return [query_results[:k] for query_results in reranked]
