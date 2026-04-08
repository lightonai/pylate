from __future__ import annotations

import logging
import os
import pickle
import shutil
from dataclasses import dataclass

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base
from .utils import convert_embeddings_to_torch

logger = logging.getLogger(__name__)


@dataclass
class WARPSearchConfig:
    """Search-time hyperparameters for the WARP index.

    All parameters default to ``None`` which lets xtr-warp auto-tune them
    based on index characteristics.

    Parameters
    ----------
    nprobe
        Number of inverted file probes during search.
    bound
        Number of centroids to consider per query.
    t_prime
        Value for the t_prime scoring policy.
    max_candidates
        Maximum candidates to consider before the final sort.
    centroid_score_threshold
        Threshold for centroid scores (0 to 1).
    batch_size
        Batch size for the query matmul against centroids.
    num_threads
        Upper bound of threads for CPU search. Ignored on CUDA.
    """

    nprobe: int | None = None
    bound: int | None = None
    t_prime: int | None = None
    max_candidates: int | None = None
    centroid_score_threshold: float | None = None
    batch_size: int = 8192
    num_threads: int | None = 1


@dataclass
class WARPIndexingConfig:
    """Index-creation and incremental-add hyperparameters for the WARP index.

    Parameters
    ----------
    nbits
        Number of bits for product quantization.
        Lower values mean more compression but can reduce accuracy.
    kmeans_niters
        Number of K-means iterations during index creation.
    max_points_per_centroid
        Maximum points per centroid for K-means.
    n_samples_kmeans
        Number of samples for K-means clustering.
        If None, defaults based on number of documents.
    use_triton
        Whether to use Triton kernels for K-means. Faster but
        non-deterministic due to race conditions.
        If None, uses Triton when available on GPU.
    seed
        Random seed for K-means reproducibility.
    min_outliers
        Minimum number of outlier embeddings required to trigger centroid
        expansion during incremental add.
    max_growth_rate
        Maximum ratio of new centroids relative to the existing codebook
        size during centroid expansion.
    compact_threshold
        Fraction of deleted passages that triggers auto-compaction (0 to 1).
        Default 0.2 means compaction runs when 20% of passages are deleted.
        Set to None to disable auto-compaction.
    """

    nbits: int = 4
    kmeans_niters: int = 4
    max_points_per_centroid: int = 256
    n_samples_kmeans: int | None = None
    use_triton: bool | None = None
    seed: int = 42
    min_outliers: int = 50
    max_growth_rate: float = 0.1
    compact_threshold: float | None = 0.2


class WARP(Base):
    """WARP index using the xtr-warp-rs backend for high-performance multi-vector search.

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the collection if it already exists.
    search
        Search hyperparameters. If None, uses defaults (auto-tuned by WARP).
    indexing
        Index-creation hyperparameters. If None, uses defaults.
    show_progress
        Whether to display progress bars during operations.
    device
        Device for computation (e.g. "cpu", "cuda", "cuda:0").
        If None, defaults to "cuda" if available, else "cpu".
    dtype
        Precision for centroids and bucket weights during search
        (e.g. torch.float32, torch.float16). Affects memory footprint.
    mmap
        Memory-map large index tensors (codes and residuals) to reduce
        memory usage. Only supported on CPU.

    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "warp",
        override: bool = False,
        search: WARPSearchConfig | None = None,
        indexing: WARPIndexingConfig | None = None,
        show_progress: bool = True,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
    ) -> None:
        try:
            from xtr_warp import search as warp_search
        except ImportError:
            raise ImportError(
                "xtr-warp-rs is not installed. Please install it with: "
                '`pip install "pylate[warp]"` or `pip install xtr-warp-rs`.'
            )

        self.index_folder = index_folder
        self.index_name = index_name
        self.search_config = search or WARPSearchConfig()
        self.indexing_config = indexing or WARPIndexingConfig()
        self.show_progress = show_progress
        self.dtype = dtype
        self.mmap = mmap

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create the index directory structure
        self.index_path = os.path.join(index_folder, index_name)
        self.warp_index_path = os.path.join(self.index_path, "warp_index")
        if override and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)

        os.makedirs(self.index_path, exist_ok=True)

        # Pickle mappings for document IDs
        self.documents_ids_to_warp_ids_path = os.path.join(
            self.index_path, "documents_ids_to_warp_ids.pkl"
        )
        self.warp_ids_to_documents_ids_path = os.path.join(
            self.index_path, "warp_ids_to_documents_ids.pkl"
        )

        # Initialize the XTRWarp instance
        self.warp = warp_search.XTRWarp(index=self.warp_index_path, device=self.device)

        # Check if index already exists on disk
        self.is_indexed = os.path.exists(self.documents_ids_to_warp_ids_path)
        if self.is_indexed:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load the WARP index into memory if not already loaded."""
        if self.warp._loaded_searchers is None:
            self.warp.load(device=self.device, dtype=self.dtype, mmap=self.mmap)

    def _load_documents_ids_to_warp_ids(self) -> dict:
        """Load the pickle file that maps document IDs to WARP passage IDs."""
        if os.path.exists(self.documents_ids_to_warp_ids_path):
            with open(self.documents_ids_to_warp_ids_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _load_warp_ids_to_documents_ids(self) -> dict:
        """Load the pickle file that maps WARP passage IDs to document IDs."""
        if os.path.exists(self.warp_ids_to_documents_ids_path):
            with open(self.warp_ids_to_documents_ids_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_mappings(
        self,
        documents_ids_to_warp_ids: dict,
        warp_ids_to_documents_ids: dict,
    ) -> None:
        """Save the ID mappings to pickle files."""
        with open(self.documents_ids_to_warp_ids_path, "wb") as f:
            pickle.dump(documents_ids_to_warp_ids, f)
        with open(self.warp_ids_to_documents_ids_path, "wb") as f:
            pickle.dump(warp_ids_to_documents_ids, f)

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        **kwargs,
    ) -> "WARP":
        """Add documents to the index.

        On the first call this creates the WARP index. Subsequent calls use
        WARP's incremental add which appends documents and may expand the
        centroid codebook if many new embeddings are outliers.

        Parameters
        ----------
        documents_ids
            Document IDs to associate with the embeddings.
        documents_embeddings
            The document embeddings to index.
        """
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        documents_embeddings_torch = convert_embeddings_to_torch(documents_embeddings)

        documents_ids_to_warp_ids = self._load_documents_ids_to_warp_ids()
        warp_ids_to_documents_ids = self._load_warp_ids_to_documents_ids()

        idx = self.indexing_config
        if not self.is_indexed:
            self.warp.create(
                embeddings_source=documents_embeddings_torch,
                device=self.device,
                kmeans_niters=idx.kmeans_niters,
                max_points_per_centroid=idx.max_points_per_centroid,
                nbits=idx.nbits,
                n_samples_kmeans=idx.n_samples_kmeans,
                seed=idx.seed,
                use_triton_kmeans=idx.use_triton,
                show_progress=self.show_progress,
            )
            warp_ids = list(range(len(documents_embeddings_torch)))
            self.is_indexed = True
        else:
            warp_ids = self.warp.add(
                embeddings_source=documents_embeddings_torch,
                reload=True,
                min_outliers=idx.min_outliers,
                max_growth_rate=idx.max_growth_rate,
                max_points_per_centroid=idx.max_points_per_centroid,
                show_progress=self.show_progress,
            )

        self._ensure_loaded()

        documents_ids_to_warp_ids.update(zip(documents_ids, warp_ids))
        warp_ids_to_documents_ids.update(zip(warp_ids, documents_ids))
        self._save_mappings(documents_ids_to_warp_ids, warp_ids_to_documents_ids)

        return self

    def remove_documents(self, documents_ids: list[str]) -> "WARP":
        """Remove documents from the index.

        Uses WARP's tombstone deletion. When the tombstone ratio exceeds
        ``compact_threshold``, WARP automatically compacts the index.

        Parameters
        ----------
        documents_ids
            The document IDs to remove.
        """
        documents_ids_to_warp_ids = self._load_documents_ids_to_warp_ids()
        warp_ids_to_documents_ids = self._load_warp_ids_to_documents_ids()

        warp_ids_to_remove = []
        for document_id in documents_ids:
            if document_id in documents_ids_to_warp_ids:
                warp_id = documents_ids_to_warp_ids[document_id]
                warp_ids_to_remove.append(warp_id)
                del documents_ids_to_warp_ids[document_id]
                del warp_ids_to_documents_ids[warp_id]

        if warp_ids_to_remove:
            self.warp.delete(
                warp_ids_to_remove,
                compact_threshold=self.indexing_config.compact_threshold,
            )

        self._save_mappings(documents_ids_to_warp_ids, warp_ids_to_documents_ids)
        return self

    def update_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
    ) -> "WARP":
        """Update document embeddings in-place, preserving passage IDs.

        More efficient than delete + add when re-indexing changed documents.

        Parameters
        ----------
        documents_ids
            The document IDs to update. Must already exist in the index.
        documents_embeddings
            The new embeddings for each document.
        """
        documents_embeddings_torch = convert_embeddings_to_torch(documents_embeddings)
        documents_ids_to_warp_ids = self._load_documents_ids_to_warp_ids()

        warp_ids = [
            documents_ids_to_warp_ids[doc_id]
            for doc_id in documents_ids
            if doc_id in documents_ids_to_warp_ids
        ]

        if warp_ids:
            self.warp.update(
                passage_ids=warp_ids,
                embeddings_source=documents_embeddings_torch,
                reload=True,
                show_progress=self.show_progress,
            )

        return self

    def compact(self) -> "WARP":
        """Compact the index, physically removing deleted passages and empty centroids.

        Use after bulk deletions to reclaim disk space.
        """
        self.warp.compact(reload=True, show_progress=self.show_progress)
        return self

    def __call__(
        self,
        queries_embeddings: np.ndarray
        | torch.Tensor
        | list[np.ndarray]
        | list[torch.Tensor],
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
            Can be a single list (same filter applied to all queries) or
            a list of lists (per-query filter; must match the number of queries).

        Returns
        -------
        List of lists containing RerankResult with 'id' and 'score' keys.
        """
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before querying."
            )

        self._ensure_loaded()

        warp_ids_to_documents_ids = self._load_warp_ids_to_documents_ids()
        documents_ids_to_warp_ids = self._load_documents_ids_to_warp_ids()

        queries_embeddings = convert_embeddings_to_torch(queries_embeddings)

        # Convert subset from document IDs to WARP passage IDs
        warp_subset = None
        if subset is not None:
            if len(subset) == 0:
                warp_subset = []
            elif isinstance(subset[0], list):
                # Per-query subsets
                warp_subset = [
                    [
                        documents_ids_to_warp_ids[doc_id]
                        for doc_id in query_subset
                        if doc_id in documents_ids_to_warp_ids
                    ]
                    for query_subset in subset
                ]
            else:
                # Shared subset for all queries
                warp_subset = [
                    documents_ids_to_warp_ids[doc_id]
                    for doc_id in subset
                    if doc_id in documents_ids_to_warp_ids
                ]

        sc = self.search_config
        search_results = self.warp.search(
            queries_embeddings=queries_embeddings,
            top_k=k,
            nprobe=sc.nprobe,
            bound=sc.bound,
            t_prime=sc.t_prime,
            max_candidates=sc.max_candidates,
            centroid_score_threshold=sc.centroid_score_threshold,
            batch_size=sc.batch_size,
            num_threads=sc.num_threads,
            subset=warp_subset,
            show_progress=self.show_progress,
        )

        results = []
        for query_results in search_results:
            query_docs = []
            seen = set()
            for warp_id, score in query_results:
                if warp_id in warp_ids_to_documents_ids:
                    doc_id = warp_ids_to_documents_ids[warp_id]
                    if doc_id not in seen:
                        seen.add(doc_id)
                        query_docs.append(RerankResult(id=doc_id, score=float(score)))
            results.append(query_docs)

        return results

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Get document embeddings by their IDs.

        Not supported — WARP stores embeddings in compressed/quantized form.
        """
        raise NotImplementedError(
            "WARP does not provide direct access to document embeddings. "
            "The embeddings are stored in compressed/quantized form and cannot "
            "be retrieved."
        )
