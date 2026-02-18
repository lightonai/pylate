from __future__ import annotations

import logging
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
    ...     name="xtr_scann",
    ...     store_embeddings=False,
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
        if isinstance(index, indexes.PLAID):
            raise ValueError(
                "XTR retriever requires token-level index outputs "
                "(`documents_ids` and `distances`). PLAID-style end-to-end indices "
                "are not supported."
            )
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
        imputation: str = "min",
        percentile: float = 10.0,
        power_law_multiplier: float = 100.0,
        return_timing: bool = False,
    ) -> list[list[RerankResult]] | tuple[list[list[RerankResult]], dict[str, float]]:
        """Retrieve documents using XTR (eXact Token Retrieval) scoring.

        XTR differs from standard ColBERT retrieval in that it doesn't do a full
        reranking step. Instead, it only scores documents using initially retrieved
        tokens and imputes missing scores based on the chosen imputation strategy.

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
        imputation
            Strategy for imputing missing scores. Options:
            - "min": Use minimum retrieved score per query token (default, original XTR).
            - "zero": Impute with zero (missing tokens contribute nothing).
            - "mean": Use mean of retrieved scores per query token.
            - "percentile": Use specified percentile of retrieved scores.
            - "power_law": Fit power-law curve to retrieved scores and extrapolate.
        percentile
            Percentile value (0-100) for percentile imputation. Default is 10.0.
        power_law_multiplier
            Multiplier for k' when extrapolating power-law. Default is 100.0.
        return_timing
            If True, return tuple of (results, timing_dict) with per-stage timing data.

        Returns
        -------
        list[list[RerankResult]]
            List of results for each query, where each result contains
            document IDs and scores sorted by score (descending).

        """
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
            total=len(queries_embeddings) // batch_size,
        )
        for batch_queries_embeddings in progress_bar:
            # Initial retrieval from index
            retrieval_start = time.time()
            index_results = self.index(batch_queries_embeddings, k=k_token)
            if not isinstance(index_results, dict):
                raise ValueError(
                    "XTR retriever expects token-level dict outputs from the index "
                    "with `documents_ids` and `distances`."
                )
            if "documents_ids" not in index_results or "distances" not in index_results:
                raise ValueError(
                    "XTR retriever received invalid index output. Expected keys: "
                    "`documents_ids` and `distances`."
                )
            token_doc_ids = index_results["documents_ids"]
            token_distances = index_results["distances"]
            if len(token_doc_ids) != len(token_distances):
                raise ValueError(
                    "XTR retriever received invalid index output. "
                    "`documents_ids` and `distances` must have the same batch length."
                )
            retrieval_time = time.time() - retrieval_start
            total_retrieval_time += retrieval_time

            # XTR scoring
            scoring_start = time.time()
            for query_doc_ids, query_scores in zip(token_doc_ids, token_distances):
                # Use the score_xtr helper function to compute XTR scores
                query_results = score_xtr(
                    query_doc_ids=query_doc_ids,
                    query_scores=query_scores,
                    k=k,
                    device=device,
                    imputation=imputation,
                    percentile=percentile,
                    power_law_multiplier=power_law_multiplier,
                )

                results.append(query_results)
            scoring_time = time.time() - scoring_start
            total_scoring_time += scoring_time
            num_batches += 1

            batch_count = max(1, len(batch_queries_embeddings))
            per_query_retrieval = retrieval_time / batch_count  # seconds per query
            per_query_scoring = scoring_time / batch_count  # seconds per query
            per_query_total = (
                per_query_retrieval + per_query_scoring
            )  # seconds per query
            if not progress_bar.disable:
                progress_bar.set_postfix(
                    {
                        "per_query_retrieval (s)": f"{per_query_retrieval:.3f} ({per_query_retrieval / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_scoring (s)": f"{per_query_scoring:.3f} ({per_query_scoring / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_total (s)": f"{per_query_total:.3f}",
                    }
                )

        total_time = total_retrieval_time + total_scoring_time
        # Log timing breakdown if verbose
        if self.verbose:
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

        if return_timing:
            num_queries = len(queries_embeddings)
            timing_dict = {
                "token_retrieval": total_retrieval_time / num_queries
                if num_queries > 0
                else 0.0,
                "score_imputation": total_scoring_time / num_queries
                if num_queries > 0
                else 0.0,
                "total_time": total_time / num_queries if num_queries > 0 else 0.0,
            }
            return results, timing_dict
        return results
