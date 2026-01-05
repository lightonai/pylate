from __future__ import annotations

import logging
import os
from bisect import bisect_left

import numpy as np
import torch
from fast_plaid import search
from sqlitedict import SqliteDict

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)


def convert_embeddings_to_torch(
    embeddings: np.ndarray | torch.Tensor | list,
) -> list[torch.Tensor]:
    """Convert embeddings to list of torch tensors as expected by fast-plaid."""
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


class FastPlaid(Base):
    """FastPlaid index using the fast-plaid backend for high-performance multi-vector search.

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
    use_triton
        Whether to use triton kernels when computing kmeans using fast-plaid. Triton kernels are faster, but yields some variance due to race condition, set to false to get 100% reproducible results. If unset, will use triton kernels if possible.

    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "fast_plaid",
        override: bool = False,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_ivf_probe: int = 8,
        n_full_scores: int = 8192,
        n_samples_kmeans: int | None = None,
        batch_size: int = 1 << 18,
        show_progress: bool = True,
        device: str | list[str] | None = None,
        use_triton: bool | None = None,
    ) -> None:
        self.index_folder = index_folder
        self.index_name = index_name
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.max_points_per_centroid = max_points_per_centroid
        self.n_ivf_probe = n_ivf_probe
        self.n_full_scores = n_full_scores
        self.n_samples_kmeans = n_samples_kmeans
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.device = device
        self.use_triton = use_triton

        # Create the index directory structure
        self.index_path = os.path.join(index_folder, index_name)
        self.fast_plaid_index_path = os.path.join(self.index_path, "fast_plaid_index")

        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        # SQLite mappings for document IDs
        self.documents_ids_to_plaid_ids_path = os.path.join(
            self.index_path, "documents_ids_to_plaid_ids.sqlite"
        )
        self.plaid_ids_to_documents_ids_path = os.path.join(
            self.index_path, "plaid_ids_to_documents_ids.sqlite"
        )

        # Initialize or load the fast-plaid index
        self.fast_plaid = search.FastPlaid(
            index=self.fast_plaid_index_path, device=device
        )

        if override:
            # Remove existing SQLite mappings
            if os.path.exists(self.documents_ids_to_plaid_ids_path):
                os.remove(self.documents_ids_to_plaid_ids_path)
            if os.path.exists(self.plaid_ids_to_documents_ids_path):
                os.remove(self.plaid_ids_to_documents_ids_path)
            self.is_indexed = False
        else:
            # Check if index already exists
            documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
            self.is_indexed = len(documents_ids_to_plaid_ids) > 0
            documents_ids_to_plaid_ids.close()

    def _load_documents_ids_to_plaid_ids(self) -> SqliteDict:
        """Load the SQLite database that maps document IDs to PLAID IDs."""
        return SqliteDict(self.documents_ids_to_plaid_ids_path, outer_stack=False)

    def _load_plaid_ids_to_documents_ids(self) -> SqliteDict:
        """Load the SQLite database that maps PLAID IDs to document IDs."""
        return SqliteDict(self.plaid_ids_to_documents_ids_path, outer_stack=False)

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        **kwargs,
    ) -> "FastPlaid":
        """Add documents to the index using fast-plaid backend."""
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        # Convert embeddings to torch tensors
        documents_embeddings_torch = convert_embeddings_to_torch(documents_embeddings)

        # Load SQLite mappings
        documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()

        # Get the current number of documents for ID assignment
        current_max_id = (
            max([int(k) for k in plaid_ids_to_documents_ids.keys()])
            if plaid_ids_to_documents_ids
            else -1
        )

        if not self.is_indexed:
            # Create new index
            self.fast_plaid.create(
                documents_embeddings=documents_embeddings_torch,
                kmeans_niters=self.kmeans_niters,
                max_points_per_centroid=self.max_points_per_centroid,
                nbits=self.nbits,
                n_samples_kmeans=self.n_samples_kmeans,
                use_triton_kmeans=self.use_triton,
            )
            plaid_ids = list(range(len(documents_embeddings_torch)))
            self.is_indexed = True
        else:
            # Update existing index
            logger.warning(
                "Adding documents to existing index. This uses fast-plaid's update method "
                "which does not recompute centroids and may result in slightly lower accuracy."
            )
            self.fast_plaid.update(documents_embeddings=documents_embeddings_torch)
            # Assign new plaid IDs starting from current_max_id + 1
            plaid_ids = list(
                range(
                    current_max_id + 1,
                    current_max_id + 1 + len(documents_embeddings_torch),
                )
            )

        # Store mappings
        for plaid_id, document_id in zip(plaid_ids, documents_ids):
            documents_ids_to_plaid_ids[document_id] = plaid_id
            plaid_ids_to_documents_ids[plaid_id] = document_id

        documents_ids_to_plaid_ids.commit()
        documents_ids_to_plaid_ids.close()

        plaid_ids_to_documents_ids.commit()
        plaid_ids_to_documents_ids.close()

        return self

    def remove_documents(self, documents_ids: list[str]) -> "FastPlaid":
        """Remove documents from the index and update ID mappings.

        Note: Fast-plaid does not support direct document removal.
        This method removes the document mappings, updates plaid IDs to maintain
        sequential ordering, and the embeddings remain in the index.
        For complete removal, consider rebuilding the index without the unwanted documents.

        Parameters
        ----------
        documents_ids
            The document IDs to remove.
        """

        documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()

        # Collect plaid_ids to remove and track which ones are being deleted
        plaid_ids_to_remove = []
        for document_id in documents_ids:
            if document_id in documents_ids_to_plaid_ids:
                plaid_id = documents_ids_to_plaid_ids[document_id]
                plaid_ids_to_remove.append(plaid_id)
                del documents_ids_to_plaid_ids[document_id]
                del plaid_ids_to_documents_ids[plaid_id]
        self.fast_plaid.delete(plaid_ids_to_remove)

        # Sort plaid_ids to remove for efficient offset calculation
        plaid_ids_to_remove.sort()

        updated_plaid_ids_to_documents_ids = {}
        for old_plaid_id, document_id in plaid_ids_to_documents_ids.items():
            old_plaid_id_int = int(old_plaid_id)
            # bisect_left gives count of elements < old_plaid_id_int
            offset = bisect_left(plaid_ids_to_remove, old_plaid_id_int)
            new_plaid_id = old_plaid_id_int - offset

            updated_plaid_ids_to_documents_ids[new_plaid_id] = document_id
            documents_ids_to_plaid_ids[document_id] = new_plaid_id

        # Replace the old mappings with updated ones
        plaid_ids_to_documents_ids.clear()
        plaid_ids_to_documents_ids.update(updated_plaid_ids_to_documents_ids)

        documents_ids_to_plaid_ids.commit()
        documents_ids_to_plaid_ids.close()

        plaid_ids_to_documents_ids.commit()
        plaid_ids_to_documents_ids.close()

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
        """Query the index for the nearest neighbors of the queries embeddings.

        Parameters
        ----------
        queries_embeddings
            The query embeddings. Can be numpy array, torch tensor, or list of numpy arrays/torch tensors.
        k
            The number of nearest neighbors to return.
        subset
            Optional subset of document IDs to restrict search to.
            Can be a single list (same filter for all queries) or
            list of lists (different filter per query).
            Document IDs should match the IDs used when adding documents.

        Returns
        -------
        List of lists containing dictionaries with 'id' and 'score' keys.
        """
        if not self.is_indexed:
            error = """
            The index is empty. Please add documents before querying.
            """
            raise ValueError(error)

        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()
        documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()

        # Convert queries to torch tensor format expected by fast-plaid
        queries_embeddings = convert_embeddings_to_torch(queries_embeddings)

        # Convert subset from document IDs to plaid IDs if provided
        plaid_subset = None
        if subset is not None:
            if len(subset) == 0:
                # Empty list - return empty results
                plaid_subset = []
            elif isinstance(subset[0], list):
                # List of lists - different subset for each query
                plaid_subset = []
                for query_subset in subset:
                    query_plaid_ids = [
                        documents_ids_to_plaid_ids[doc_id]
                        for doc_id in query_subset
                        if doc_id in documents_ids_to_plaid_ids
                    ]
                    plaid_subset.append(query_plaid_ids)
            else:
                # Single list - same subset for all queries
                plaid_subset = [
                    documents_ids_to_plaid_ids[doc_id]
                    for doc_id in subset
                    if doc_id in documents_ids_to_plaid_ids
                ]

        # Perform search using fast-plaid
        search_results = self.fast_plaid.search(
            queries_embeddings=queries_embeddings,
            top_k=k,
            batch_size=self.batch_size,
            n_ivf_probe=self.n_ivf_probe,
            n_full_scores=self.n_full_scores,
            show_progress=self.show_progress,
            subset=plaid_subset,
        )

        # Convert results to expected format
        results = []
        for query_results in search_results:
            query_docs = []
            for plaid_id, score in query_results:
                if plaid_id in plaid_ids_to_documents_ids:
                    doc_id = plaid_ids_to_documents_ids[plaid_id]
                    query_docs.append(RerankResult(id=doc_id, score=float(score)))
            results.append(query_docs)

        plaid_ids_to_documents_ids.close()
        documents_ids_to_plaid_ids.close()
        return results

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Get document embeddings by their IDs.

        Note: Fast-plaid does not provide direct access to document embeddings.
        This method is not implemented as the embeddings are stored in compressed form.
        """
        raise NotImplementedError(
            "Fast-plaid does not provide direct access to document embeddings. "
            "The embeddings are stored in compressed/quantized form and cannot be retrieved."
        )
