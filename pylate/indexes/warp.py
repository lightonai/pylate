from __future__ import annotations

import logging
import os
import pickle
import shutil

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base
from .utils import convert_embeddings_to_torch

logger = logging.getLogger(__name__)


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
    nbits
        The number of bits to use for product quantization.
        Lower values mean more compression and potentially faster searches but
        can reduce accuracy.
    kmeans_niters
        The number of iterations for the K-means algorithm used during index
        creation. This influences the quality of the initial centroid
        assignments.
    max_points_per_centroid
        The maximum number of points (token embeddings) that can be assigned to
        a single centroid during K-means. Helps balance the clusters.
    n_samples_kmeans
        The number of samples to use for K-means clustering. If None, defaults
        to a value chosen by xtr-warp based on the number of documents.
    seed
        Random seed for K-means reproducibility.
    use_triton
        Whether to use Triton kernels when computing K-means. Triton kernels are
        faster but yield some variance due to race conditions; set to False for
        100% reproducible results. If None, uses Triton when available on GPU.
    min_outliers
        Minimum number of outlier embeddings required to trigger centroid
        expansion during incremental ``add_documents`` calls.
    max_growth_rate
        Maximum ratio of new centroids relative to the existing codebook size
        during centroid expansion on incremental adds.
    n_ivf_probe
        The number of inverted file list probes to perform during search. This
        parameter controls the number of clusters to search within the index
        for each query. Higher values improve recall but increase search time.
        Same parameter as `n_ivf_probe` on `indexes.PLAID`. If None, xtr-warp
        auto-tunes based on index characteristics.
    bound
        Number of centroids to consider per query token. If None, auto-tuned.
    t_prime
        Value for the t_prime scoring policy. If None, auto-tuned.
    max_candidates
        Maximum number of candidate documents to consider before the final
        sort. If None, auto-tuned.
    centroid_score_threshold
        Threshold on centroid scores (between 0 and 1) used to prune candidates
        during search. If None, auto-tuned.
    batch_size
        The internal batch size used when computing the query × centroids
        matmul during search.
    num_threads
        Upper bound on threads for CPU search. Ignored on CUDA.
    show_progress
        If set to True, a progress bar is displayed during indexing and search
        operations.
    device
        Device for computation (e.g. "cpu", "cuda", "cuda:0"). If None,
        defaults to "cuda" when available, else "cpu".
    dtype
        Precision used for centroids and bucket weights when the index is
        loaded for search (e.g. ``torch.float32``, ``torch.float16``). Affects
        memory footprint and search speed.
    mmap
        Memory-map large index tensors (codes and residuals) to reduce memory
        usage. Only supported on CPU.

    """

    is_end_to_end_index = True

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "warp",
        override: bool = False,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        use_triton: bool | None = None,
        min_outliers: int = 50,
        max_growth_rate: float = 0.1,
        n_ivf_probe: int | None = 32,
        bound: int | None = None,
        t_prime: int | None = 100_000,
        max_candidates: int | None = None,
        centroid_score_threshold: float | None = None,
        batch_size: int = 8192,
        num_threads: int | None = 1,
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

        # Indexing hyperparameters
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.max_points_per_centroid = max_points_per_centroid
        self.n_samples_kmeans = n_samples_kmeans
        self.seed = seed
        self.use_triton = use_triton
        self.min_outliers = min_outliers
        self.max_growth_rate = max_growth_rate

        # Search hyperparameters
        self.n_ivf_probe = n_ivf_probe
        self.bound = bound
        self.t_prime = t_prime
        self.max_candidates = max_candidates
        self.centroid_score_threshold = centroid_score_threshold
        self.batch_size = batch_size
        self.num_threads = num_threads

        # Runtime / loading
        self.show_progress = show_progress
        self.dtype = dtype
        self.mmap = mmap
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

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
        self._loaded = False

        # Check if index already exists on disk
        self.is_indexed = os.path.exists(self.documents_ids_to_warp_ids_path)
        if self.is_indexed:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load the WARP index into memory if not already loaded."""
        if not self._loaded:
            self.warp.load(device=self.device, dtype=self.dtype, mmap=self.mmap)
            self._loaded = True

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
        **kwargs
            Accepted for compatibility with the base ``Index`` interface
            (e.g. ``batch_size``). Ignored by WARP, which manages batching
            internally.
        """
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        documents_embeddings_torch = convert_embeddings_to_torch(documents_embeddings)

        documents_ids_to_warp_ids = self._load_documents_ids_to_warp_ids()
        warp_ids_to_documents_ids = self._load_warp_ids_to_documents_ids()

        if not self.is_indexed:
            self.warp.create(
                embeddings_source=documents_embeddings_torch,
                device=self.device,
                kmeans_niters=self.kmeans_niters,
                max_points_per_centroid=self.max_points_per_centroid,
                nbits=self.nbits,
                n_samples_kmeans=self.n_samples_kmeans,
                seed=self.seed,
                use_triton_kmeans=self.use_triton,
                show_progress=self.show_progress,
            )
            warp_ids = list(range(len(documents_embeddings_torch)))
            self.is_indexed = True
        else:
            warp_ids = self.warp.add(
                embeddings_source=documents_embeddings_torch,
                reload=True,
                min_outliers=self.min_outliers,
                max_growth_rate=self.max_growth_rate,
                max_points_per_centroid=self.max_points_per_centroid,
                show_progress=self.show_progress,
            )

        self._ensure_loaded()

        documents_ids_to_warp_ids.update(zip(documents_ids, warp_ids))
        warp_ids_to_documents_ids.update(zip(warp_ids, documents_ids))
        self._save_mappings(documents_ids_to_warp_ids, warp_ids_to_documents_ids)

        return self

    def remove_documents(self, documents_ids: list[str]) -> "WARP":
        """Remove documents from the index.

        Uses WARP's tombstone deletion followed by an immediate compaction so
        that disk space is reclaimed and tombstoned passages are physically
        removed on every call.

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
            self.warp.delete(warp_ids_to_remove)
            self.warp.compact(reload=True, show_progress=self.show_progress)

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

        search_results = self.warp.search(
            queries_embeddings=queries_embeddings,
            top_k=k,
            nprobe=self.n_ivf_probe,
            bound=self.bound,
            t_prime=self.t_prime,
            max_candidates=self.max_candidates,
            centroid_score_threshold=self.centroid_score_threshold,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
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
