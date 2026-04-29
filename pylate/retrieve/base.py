from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import tqdm

from ..indexes.base import Base as BaseIndex
from ..rank import RerankResult
from ..utils import iter_batch

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Common scaffolding for token-index retrievers.

    Subclasses implement :meth:`_score_batch` to convert a batch's token-level
    index hits into ranked ``RerankResult`` lists. The base owns input
    normalization, device defaulting, end-to-end short-circuit, and batching.
    """

    # Per-token candidates pulled from the index when the token path is used.
    default_k_token: int = 100
    default_batch_size: int = 50
    progress_desc: str = "Retrieving documents"

    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Retrieve documents for a list of queries.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of token-level candidates to retrieve from the index
            before scoring. Defaults to ``default_k_token`` for this retriever.
        device
            Device used for the scoring step. Defaults to the queries
            embeddings device when available, otherwise ``"cpu"``.
        batch_size
            Query batch size. Defaults to ``default_batch_size``.
        subset
            Optional document-id filter. End-to-end indexes receive it
            directly; on the token path, subclasses decide via
            :meth:`_validate_subset_token_path`.

        """
        k_token = self.default_k_token if k_token is None else k_token
        batch_size = self.default_batch_size if batch_size is None else batch_size

        # End-to-end indexes (e.g. PLAID) handle scoring internally and return
        # RerankResult directly.
        if self.index.is_end_to_end_index:
            return self.index(
                queries_embeddings=queries_embeddings,
                k=k,
                subset=subset,
            )

        self._validate_subset_token_path(subset)

        # Single-query input: a 2D array/tensor is one query of shape
        # (num_tokens, dim), not num_tokens queries — wrap in a list so the
        # batch loop sees a single element.
        if isinstance(queries_embeddings, (np.ndarray, torch.Tensor)):
            if queries_embeddings.ndim == 2:
                queries_embeddings = [queries_embeddings]

        if device is None:
            if queries_embeddings and isinstance(queries_embeddings[0], torch.Tensor):
                device = str(queries_embeddings[0].device)
            else:
                device = "cpu"

        if k > k_token:
            logger.warning(
                f"k ({k}) is greater than k_token ({k_token}), setting k_token to k."
            )
            k_token = k

        results: list[list[RerankResult]] = []
        progress_bar = tqdm.tqdm(
            iter_batch(queries_embeddings, batch_size=batch_size, tqdm_bar=False),
            desc=f"{self.progress_desc} (bs={batch_size})",
            disable=not self._show_progress(),
            total=math.ceil(len(queries_embeddings) / batch_size),
        )
        for batch_queries_embeddings in progress_bar:
            hits = self.index(
                queries_embeddings=batch_queries_embeddings,
                k=k_token,
            )
            results.extend(
                self._score_batch(
                    batch_queries_embeddings, hits, k=k, device=device
                )
            )
        return results

    def _validate_subset_token_path(
        self, subset: list[list[str]] | list[str] | None
    ) -> None:
        """Hook for subclasses. Override to raise if ``subset`` cannot be
        honored on the token-path scoring branch. The default is a no-op
        (the argument is silently ignored on the token path)."""

    def _show_progress(self) -> bool:
        """Whether to render the per-batch tqdm bar. Defaults to ``True``;
        subclasses can override to expose a ``verbose`` knob."""
        return True

    @abstractmethod
    def _score_batch(
        self,
        batch_queries_embeddings: list | np.ndarray | torch.Tensor,
        hits: dict,
        *,
        k: int,
        device: str,
    ) -> list[list[RerankResult]]:
        """Convert one batch of index hits into ranked ``RerankResult`` lists."""
