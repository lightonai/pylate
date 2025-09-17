from __future__ import annotations

import logging
import warnings

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)


class PLAID(Base):
    """PLAID index with choice between fast-plaid (Rust-based) and Stanford NLP backends.

    This class provides a unified interface for PLAID indexing that can use either:
    - FastPlaid: High-performance Rust-based implementation (default)
    - Stanford PLAID: Original Stanford NLP implementation (deprecated)

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the collection if it already exists.
    use_fast
        If True (default), use fast-plaid backend. If False, use Stanford PLAID backend.
    nbits
        The number of bits to use for product quantization.
        Lower values mean more compression and potentially faster searches but can reduce accuracy.
    kmeans_niters
        The number of iterations for the K-means algorithm used during index creation.
        This influences the quality of the initial centroid assignments.
    max_points_per_centroid
        The maximum number of points (token embeddings) that can be assigned to a single centroid during K-means.
        This helps in balancing the clusters.
    n_ivf_probe
        The number of inverted file list "probes" to perform during the search.
        This parameter controls the number of clusters to search within the index for each query.
        Higher values improve recall but increase search time.
    n_full_scores
        The number of candidate documents for which full (re-ranked) scores are computed.
        This is a crucial parameter for accuracy; higher values lead to more accurate results but increase computation.
    n_samples_kmeans
        The number of samples to use for K-means clustering.
        If None, it defaults to a value based on the number of documents.
        This parameter can be adjusted to balance between speed, memory usage and clustering quality.
    batch_size
        The internal batch size used for processing queries.
        A larger batch size might improve throughput on powerful GPUs but can consume more memory.
    show_progress
        If set to True, a progress bar will be displayed during search operations.
    device
        Specifies the device(s) to use for computation.
        If None (default) and CUDA is available, it defaults to "cuda".
        If CUDA is not available, it defaults to "cpu".
        Can be a single device string (e.g., "cuda:0" or "cpu").
        Can be a list of device strings (e.g., ["cuda:0", "cuda:1"]).
    **kwargs
        Additional arguments. Stanford PLAID specific parameters (embedding_size, nranks,
        index_bsize, ndocs, centroid_score_threshold, ncells, search_batch_size) are
        deprecated and will show warnings when used.

    Examples
    --------
    >>> from pylate import indexes, models

    >>> index = indexes.PLAID(
    ...    index_folder="test_index",
    ...    index_name="plaid_colbert",
    ...    override=True,
    ... )
    ✅ Index with FastPlaid backend.

    >>> model = models.ColBERT(
    ...    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    ... )

    >>> documents_embeddings = model.encode([
    ...    "Document content here...",
    ...    "Another document...",
    ... ] * 10, is_query=False)

    >>> index = index.add_documents(
    ...    documents_ids=range(len(documents_embeddings)),
    ...    documents_embeddings=documents_embeddings
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["search query", "hello world"],
    ...     is_query=True,
    ... )

    >>> scores = index(
    ...     queries_embeddings,
    ...     k=10,
    ... )

    >>> index = index.add_documents(
    ...    documents_ids=range(len(documents_embeddings), len(documents_embeddings) * 2),
    ...    documents_embeddings=documents_embeddings
    ... )

    >>> scores = index(
    ...     queries_embeddings,
    ...     k=25,
    ... )

    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "colbert",
        override: bool = False,
        use_fast: bool = True,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_ivf_probe: int = 8,
        n_full_scores: int = 8192,
        n_samples_kmeans: int | None = None,
        batch_size: int = 1 << 18,
        show_progress: bool = True,
        device: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        self.use_fast = use_fast
        # Check for deprecated Stanford PLAID parameters in kwargs and warn
        # These parameters will be used only if use_fast=False
        stanford_params = [
            "embedding_size",
            "nranks",
            "index_bsize",
            "ndocs",
            "centroid_score_threshold",
            "ncells",
            "search_batch_size",
        ]
        deprecated_params_found = [
            param for param in stanford_params if param in kwargs
        ]

        if deprecated_params_found and use_fast:
            message = f"""
            The use_fast=True option uses the FastPlaid backend, which ignores the
            following parameters:
            {", ".join(deprecated_params_found)}.
            """
            warnings.warn(
                message,
                FutureWarning,
                stacklevel=2,
            )

        if use_fast:
            print("✅ Index with FastPlaid backend.")
            from .fast_plaid import FastPlaid

            self._index = FastPlaid(
                index_folder=index_folder,
                index_name=index_name,
                override=override,
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                max_points_per_centroid=max_points_per_centroid,
                n_ivf_probe=n_ivf_probe,
                n_full_scores=n_full_scores,
                n_samples_kmeans=n_samples_kmeans,
                batch_size=batch_size,
                show_progress=show_progress,
                device=device,
            )
        else:
            print("📚 Index with Stanford backend.")
            from .stanford_plaid import StanfordPLAID

            # Extract Stanford PLAID parameters from kwargs with defaults
            embedding_size = kwargs.get("embedding_size", 128)
            nranks = kwargs.get("nranks", 1)
            index_bsize = kwargs.get("index_bsize", 1)
            ndocs = kwargs.get("ndocs", n_full_scores)
            centroid_score_threshold = kwargs.get("centroid_score_threshold", 0.35)
            ncells = kwargs.get("ncells", n_ivf_probe)
            search_batch_size = kwargs.get("search_batch_size", batch_size)

            self._index = StanfordPLAID(
                index_folder=index_folder,
                index_name=index_name,
                override=override,
                embedding_size=embedding_size,
                nbits=nbits,
                nranks=nranks,
                kmeans_niters=kmeans_niters,
                index_bsize=index_bsize,
                ndocs=ndocs,
                centroid_score_threshold=centroid_score_threshold,
                ncells=ncells,
                search_batch_size=search_batch_size,
            )

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        **kwargs,
    ) -> "PLAID":
        """Add documents to the index."""
        return self._index.add_documents(
            documents_ids=documents_ids,
            documents_embeddings=documents_embeddings,
            **kwargs,
        )

    def remove_documents(self, documents_ids: list[str]) -> "PLAID":
        """Remove documents from the index.

        Parameters
        ----------
        documents_ids
            The document IDs to remove.
        """
        self._index.remove_documents(documents_ids)
        return self

    def __call__(
        self,
        queries_embeddings: np.ndarray | torch.Tensor,
        k: int = 10,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Query the index for the nearest neighbors of the query embeddings.

        Parameters
        ----------
        queries_embeddings
            The query embeddings.
        k
            The number of nearest neighbors to return.
        subset
            Optional subset of document IDs to restrict search to.
            Can be a single list (same filter for all queries) or
            list of lists (different filter per query).
            Document IDs should match the IDs used when adding documents.
            Only supported with FastPlaid backend (use_fast=True).

        Returns
        -------
        List of lists containing dictionaries with 'id' and 'score' keys.

        Raises
        ------
        ValueError
            If subset is provided but Stanford PLAID backend is being used.
        """
        if subset is not None and not self.use_fast:
            raise ValueError(
                "The 'subset' parameter is only supported with FastPlaid backend. "
                "Set use_fast=True to use subset filtering."
            )

        if self.use_fast:
            return self._index(
                queries_embeddings,
                k=k,
                subset=subset,
            )

        return self._index(queries_embeddings, k=k)

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Get document embeddings by their IDs."""
        return self._index.get_documents_embeddings(document_ids)
