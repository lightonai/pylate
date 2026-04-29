from __future__ import annotations

import numpy as np
import torch

from ..indexes.base import Base as BaseIndex
from ..rank import RerankResult, score_xtr
from .base import BaseRetriever


class XTR(BaseRetriever):
    """XTR retriever.

    Performs XTR (eXact Token Retrieval) scoring: documents are scored only
    from their initially retrieved tokens, and missing token scores are filled
    in via min imputation. Differs from :class:`ColBERT`, which does a full
    MaxSim rerank using cached document embeddings.

    Parameters
    ----------
    index
        The index to use for retrieval.
    verbose
        Show a progress bar during retrieval.

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

    default_k_token = 10_000
    default_batch_size = 1
    progress_desc = "Retrieving documents (XTR)"

    def __init__(self, index: BaseIndex, verbose: bool = False) -> None:
        super().__init__(index=index)
        self.verbose = verbose

    def _show_progress(self) -> bool:
        return self.verbose

    def _validate_subset_token_path(
        self, subset: list[list[str]] | list[str] | None
    ) -> None:
        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not implemented for XTR retrieval yet."
            )

    def _score_batch(
        self,
        batch_queries_embeddings: list | np.ndarray | torch.Tensor,
        hits: dict,
        *,
        k: int,
        device: str,
    ) -> list[list[RerankResult]]:
        return [
            score_xtr(
                query_doc_ids=query_doc_ids,
                query_scores=query_scores,
                k=k,
                device=device,
            )
            for query_doc_ids, query_scores in zip(
                hits["documents_ids"], hits["distances"]
            )
        ]
