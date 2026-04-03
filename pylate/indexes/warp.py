from __future__ import annotations

import logging
import os
import pickle
import shutil

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)


def convert_embeddings_to_torch(
    embeddings: np.ndarray | torch.Tensor | list,
) -> list[torch.Tensor]:
    """Convert embeddings to list of torch tensors as expected by WARP."""
    if isinstance(embeddings, list):
        if len(embeddings) == 0:
            return []
        if isinstance(embeddings[0], torch.Tensor):
            return embeddings
        elif isinstance(embeddings[0], np.ndarray):
            return [torch.from_numpy(emb) for emb in embeddings]

    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [torch.from_numpy(embeddings[i]) for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [torch.from_numpy(embeddings)]

    if isinstance(embeddings, torch.Tensor):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [embeddings[i] for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [embeddings]

    return embeddings


class WARP(Base):
    """WARP index using the xtr-warp-rs backend for high-performance multi-vector search.

    WARP uses tombstone-based deletion: removed documents are marked as deleted and
    filtered out during search, but their storage is not reclaimed until compaction.
    This means passage IDs remain stable after deletion, simplifying ID management.

    Search hyperparameters (nprobe, bound, t_prime, max_candidates,
    centroid_score_threshold) are automatically tuned by WARP based on index
    characteristics when left as None.

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the collection if it already exists.
    nbits
        Number of bits for product quantization.
        Lower values mean more compression but can reduce accuracy.
    kmeans_niters
        Number of K-means iterations during index creation.
    max_points_per_centroid
        Maximum points per centroid for K-means.
    nprobe
        Number of inverted file probes during search.
        If None, automatically tuned based on index characteristics.
    bound
        Number of centroids to consider per query during search.
        If None, automatically tuned.
    t_prime
        Value for the t_prime scoring policy.
        If None, automatically tuned.
    max_candidates
        Maximum candidates to consider before the final sort.
        If None, automatically tuned.
    centroid_score_threshold
        Threshold for centroid scores (0 to 1).
        If None, automatically tuned.
    n_samples_kmeans
        Number of samples for K-means clustering.
        If None, defaults based on number of documents.
    batch_size
        Batch size for the query matmul against centroids.
    show_progress
        Whether to display progress bars during operations.
    device
        Device for computation (e.g. "cpu", "cuda", "cuda:0").
        If None, defaults to "cuda" if available, else "cpu".
    use_triton
        Whether to use Triton kernels for K-means. Faster but non-deterministic
        due to race conditions. If None, uses Triton when available on GPU.

    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "warp",
        override: bool = False,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nprobe: int | None = None,
        bound: int | None = None,
        t_prime: int | None = None,
        max_candidates: int | None = None,
        centroid_score_threshold: float | None = None,
        n_samples_kmeans: int | None = None,
        batch_size: int = 8192,
        show_progress: bool = True,
        device: str | None = None,
        use_triton: bool | None = None,
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
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.max_points_per_centroid = max_points_per_centroid
        self.nprobe = nprobe
        self.bound = bound
        self.t_prime = t_prime
        self.max_candidates = max_candidates
        self.centroid_score_threshold = centroid_score_threshold
        self.n_samples_kmeans = n_samples_kmeans
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.use_triton = use_triton

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
        self.warp = warp_search.XTRWarp(
            index=self.warp_index_path, device=self.device
        )

        # Check if index already exists on disk
        self.is_indexed = os.path.exists(self.documents_ids_to_warp_ids_path)
        if self.is_indexed:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load the WARP index into memory if not already loaded."""
        if self.warp._loaded_searchers is None:
            self.warp.load(device=self.device)

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

        if not self.is_indexed:
            self.warp.create(
                embeddings_source=documents_embeddings_torch,
                device=self.device,
                kmeans_niters=self.kmeans_niters,
                max_points_per_centroid=self.max_points_per_centroid,
                nbits=self.nbits,
                n_samples_kmeans=self.n_samples_kmeans,
                use_triton_kmeans=self.use_triton,
                show_progress=self.show_progress,
            )
            warp_ids = list(range(len(documents_embeddings_torch)))
            self.is_indexed = True
        else:
            # WARP's add returns the newly assigned passage IDs and handles
            # reload internally when the index was already loaded.
            warp_ids = self.warp.add(
                embeddings_source=documents_embeddings_torch,
                reload=True,
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

        Uses WARP's tombstone deletion — passage IDs remain stable and deleted
        documents are filtered out during search without requiring ID remapping.

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
            # compact_threshold=None disables auto-compaction so that passage
            # IDs stay stable and our mapping layer doesn't need to remap.
            self.warp.delete(warp_ids_to_remove, compact_threshold=None)

        self._save_mappings(documents_ids_to_warp_ids, warp_ids_to_documents_ids)
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
            Can be a single list (same filter for all queries) or
            list of lists (different filter per query).
            Note: WARP supports a single global subset per search call.
            When per-query lists are provided they are merged into their union.

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
                # Per-query subsets — WARP only supports a single subset list,
                # so merge into the union of all per-query subsets.
                all_ids = set()
                for query_subset in subset:
                    for doc_id in query_subset:
                        if doc_id in documents_ids_to_warp_ids:
                            all_ids.add(documents_ids_to_warp_ids[doc_id])
                warp_subset = list(all_ids)
            else:
                warp_subset = [
                    documents_ids_to_warp_ids[doc_id]
                    for doc_id in subset
                    if doc_id in documents_ids_to_warp_ids
                ]

        search_results = self.warp.search(
            queries_embeddings=queries_embeddings,
            top_k=k,
            nprobe=self.nprobe,
            bound=self.bound,
            t_prime=self.t_prime,
            max_candidates=self.max_candidates,
            centroid_score_threshold=self.centroid_score_threshold,
            batch_size=self.batch_size,
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
                        query_docs.append(
                            RerankResult(id=doc_id, score=float(score))
                        )
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
