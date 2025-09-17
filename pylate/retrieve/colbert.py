from __future__ import annotations

import logging

import numpy as np
import torch

from ..indexes import PLAID, Voyager
from ..rank import RerankResult, rerank
from ..utils import iter_batch

logger = logging.getLogger(__name__)


class ColBERT:
    """ColBERT retriever.

    Parameters
    ----------
    index:
        The index to use for retrieval.

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
    ✅ Index with FastPlaid backend.

    >>> index = index.add_documents(
    ...     documents_ids=documents_ids,
    ...     documents_embeddings=documents_embeddings,
    ... )

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

    def __init__(self, index: Voyager | PLAID) -> None:
        self.index = index

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int = 100,
        device: str | None = None,
        batch_size: int = 50,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]] | None:
        """Retrieve documents for a list of queries.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of documents to retrieve from the index. Defaults to `k`.
        device
            The device to use for the embeddings. Defaults to queries_embeddings device.
        batch_size
            The batch size to use for retrieval.
        subset
            Optional subset of document IDs to restrict search to.
            Can be a single list (same filter for all queries) or
            list of lists (different filter per query).
            Document IDs should match the IDs used when adding documents.
            Only supported with PLAID index.

        """
        # PLAID index directly retrieves the documents
        if isinstance(self.index, PLAID) or not isinstance(self.index, Voyager):
            return self.index(
                queries_embeddings=queries_embeddings,
                k=k,
                subset=subset,
            )

        # Other indexes first generate candidates by calling the index and then rerank them
        if k > k_token:
            logger.warning(
                f"k ({k}) is greater than k_token ({k_token}), setting k_token to k."
            )
            k_token = k
        reranking_results = []
        for queries_embeddings_batch in iter_batch(
            queries_embeddings,
            batch_size=batch_size,
            desc=f"Retrieving documents (bs={batch_size})",
        ):
            retrieved_elements = self.index(
                queries_embeddings=queries_embeddings_batch,
                k=k_token,
            )

            documents_ids = [
                list(
                    set(
                        [
                            document_id
                            for query_token_document_ids in query_documents_ids
                            for document_id in query_token_document_ids
                        ]
                    )
                )
                for query_documents_ids in retrieved_elements["documents_ids"]
            ]

            documents_embeddings = self.index.get_documents_embeddings(documents_ids)

            reranking_results.extend(
                rerank(
                    documents_ids=documents_ids,
                    queries_embeddings=queries_embeddings_batch,
                    documents_embeddings=documents_embeddings,
                    device=device,
                )
            )
        return [query_results[:k] for query_results in reranking_results]
