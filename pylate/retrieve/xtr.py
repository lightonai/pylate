from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .. import indexes
from ..rank import RerankResult, score_xtr
from ..utils import iter_batch

logger = logging.getLogger(__name__)


class XTR:
    """XTR retriever.

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

    >>> index = indexes.ScaNN(
    ...     override=True,
    ...     index_name="xtr_scann",
    ...     store_embeddings=False,
    ... )

    >>> index = index.add_documents(
    ...     documents_ids=documents_ids,
    ...     documents_embeddings=documents_embeddings,
    ...     batch_size=1,
    ... )

    >>> retriever = retrieve.XTR(index=index)

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

    """

    def __init__(self, index: indexes.Base, verbose: bool = False) -> None:
        if index.is_end_to_end_index:
            raise ValueError("XTR requires to use a non end-to-end index")
        self.index = index
        self.verbose = verbose

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int = 10_000,
        device: str = "cpu",
        batch_size: int = 1,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Retrieve documents using XTR (eXact Token Retrieval) scoring.

        XTR differs from standard ColBERT retrieval in that it doesn't do a full
        reranking step. Instead, it only scores documents using initially retrieved
        tokens and imputes missing scores using min imputation.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of documents to retrieve from the index per query token.
        device
            The device to use for XTR scoring computation. Defaults to 'cpu'.
        batch_size
            The batch size to use for retrieval.
        subset
            Optional subset of document IDs to restrict search to.
            Only supported with certain index types.


        Returns
        -------
        list[list[RerankResult]]
            List of results for each query, where each result contains
            document IDs and scores sorted by score (descending).

        """
        # Handle single query: if a 2D array/tensor is passed (num_tokens, dim),
        # wrap it in a list so it's treated as one query, not num_tokens queries.
        if isinstance(queries_embeddings, np.ndarray):
            if queries_embeddings.ndim == 2:
                queries_embeddings = [queries_embeddings]
        elif isinstance(queries_embeddings, torch.Tensor):
            if queries_embeddings.ndim == 2:
                queries_embeddings = [queries_embeddings]

        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not implemented for XTR retrieval yet."
            )

        total_retrieval_time = 0.0
        total_scoring_time = 0.0
        num_batches = 0

        results = []

        progress_bar = tqdm(
            iter_batch(queries_embeddings, batch_size=batch_size, tqdm_bar=False),
            desc="Retrieving documents (XTR)",
            disable=not self.verbose,
            total=math.ceil(len(queries_embeddings) / batch_size),
        )
        for batch_queries_embeddings in progress_bar:
            # Initial retrieval from index
            retrieval_start = time.perf_counter()
            index_results = self.index(batch_queries_embeddings, k=k_token)
            token_doc_ids = index_results["documents_ids"]
            token_distances = index_results["distances"]
            retrieval_time = time.perf_counter() - retrieval_start
            total_retrieval_time += retrieval_time

            # XTR scoring
            scoring_start = time.perf_counter()
            for query_doc_ids, query_scores in zip(token_doc_ids, token_distances):
                # Use the score_xtr helper function to compute XTR scores
                query_results = score_xtr(
                    query_doc_ids=query_doc_ids,
                    query_scores=query_scores,
                    k=k,
                    device=device,
                )

                results.append(query_results)
            scoring_time = time.perf_counter() - scoring_start
            total_scoring_time += scoring_time
            num_batches += 1

            if not progress_bar.disable:
                batch_count = max(1, len(batch_queries_embeddings))
                # seconds per query
                per_query_retrieval = retrieval_time / batch_count
                per_query_scoring = scoring_time / batch_count
                per_query_total = per_query_retrieval + per_query_scoring

                progress_bar.set_postfix(
                    {
                        "per_query_retrieval (s)": f"{per_query_retrieval:.3f} ({per_query_retrieval / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_scoring (s)": f"{per_query_scoring:.3f} ({per_query_scoring / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_total (s)": f"{per_query_total:.3f}",
                    }
                )

        # Log timing breakdown if verbose
        if self.verbose:
            total_time = total_retrieval_time + total_scoring_time

            denom = max(total_time, 1e-12)
            logger.info(
                f"XTR retrieval timing breakdown (total: {total_time:.4f}s, {num_batches} batches of {batch_size} queries):"
            )
            logger.info(
                f"  - Index retrieval: {total_retrieval_time:.4f}s ({total_retrieval_time / denom * 100:.1f}%)"
            )
            logger.info(
                f"  - XTR scoring:      {total_scoring_time:.4f}s ({total_scoring_time / denom * 100:.1f}%)"
            )
            if len(queries_embeddings) > 0:
                logger.info(
                    f"  - Per query:        {total_time / len(queries_embeddings) * 1000:.2f}ms"
                )

        return results
