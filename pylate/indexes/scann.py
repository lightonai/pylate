from __future__ import annotations

import gc
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import Base
from .utils import log_memory, reshape_embeddings

logger = logging.getLogger(__name__)


class ScaNN(Base):
    """ScaNN index. The ScaNN index is a fast and efficient index for approximate nearest neighbor search.

    **Important Notes:**
    - ScaNN is an **approximate** nearest neighbor search (not exact), designed for large-scale datasets
    - For ColBERT retrieval, PLAID is typically faster and more accurate as it's optimized for ColBERT scoring
    - ScaNN is CPU-only (no GPU acceleration)
    - Parameters are auto-tuned based on dataset size if not specified

    To use this index, you need to install the `scann` extra:

    ```bash
    pip install "pylate[scann]"
    ```

    or install scann directly:

    ```bash
    pip install scann
    ```

    Parameters
    ----------
    name
        The name of the index collection.
    embedding_size
        The number of dimensions of the embeddings.
    num_neighbors
        The number of neighbors to use for the ScaNN searcher.
    num_leaves
        The number of leaves in the ScaNN tree. If None, auto-tuned based on dataset size.
        For small datasets (<100K vectors), fewer leaves are used for speed.
    num_leaves_to_search
        The number of leaves to search during query time. If None, auto-tuned based on dataset size.
        Higher values improve recall but slow down search.
    dimensions_per_block
        The number of dimensions to use for each block. If None, auto-tuned based on dataset size.
        Defaults to 2.
    anisotropic_quantization_threshold
        The threshold for anisotropic quantization. If None, auto-tuned based on dataset size.
        Defaults to 0.2.
    training_sample_size
        The number of samples to use for training the ScaNN index.
    verbose
        Verbosity configuration:
        - ``False`` or ``"none"``: disable logs
        - ``True`` or ``"init"``: log build/load/indexing only
        - ``"all"``: log build/load/indexing and per-query retrieval
    verbose_level
        Backward-compatible alias for verbosity scope (``"none"``, ``"init"``,
        ``"all"``). If set, it overrides ``verbose``.
    use_autopilot
        Whether to use ScaNN's autopilot() method for automatic parameter tuning.
        If True, overrides num_leaves, num_leaves_to_search, and training_sample_size.
        Defaults to False.
    store_embeddings
        Whether to store the embeddings in the index. If True, the embeddings will be stored in the index.
        Defaults to True. This is required to use the get_documents_embeddings method.
    index_folder
        The folder where the index will be saved/loaded. If None, indices are not persisted to disk.
        Defaults to None.
    override
        Whether to override the index if it already exists. If False and index exists, it will be loaded.
        Defaults to False.

    """

    def __init__(
        self,
        name: str | None = "ScaNN_index",
        embedding_size: int = 128,
        num_neighbors: int | None = 10,
        num_leaves: int | None = None,
        num_leaves_to_search: int | None = None,
        dimensions_per_block: int | None = 2,
        anisotropic_quantization_threshold: float | None = 0.2,
        training_sample_size: int | None = None,
        verbose: bool | str = "none",
        use_autopilot: bool = False,
        store_embeddings: bool = True,
        index_folder: str | None = None,
        override: bool = False,
        verbose_level: str | None = None,
    ) -> None:
        self.name = name
        self.embedding_size = embedding_size
        self.num_neighbors = num_neighbors
        self.verbose_level = (
            verbose_level
            if verbose_level is not None
            else self._normalize_verbose(verbose=verbose)
        )
        if self.verbose_level not in {"none", "init", "all"}:
            raise ValueError(
                f"Invalid verbosity level: {self.verbose_level}. "
                "Expected one of: 'none', 'init', 'all'."
            )
        self.verbose = self.verbose_level in ("init", "all")
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.dimensions_per_block = dimensions_per_block
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.training_sample_size = training_sample_size
        self.use_autopilot = use_autopilot
        self.store_embeddings = store_embeddings
        self.index_folder = index_folder
        self.override = override
        
        # In-memory data structures
        self.searcher = None
        # Note: embedding_id == position (sequential IDs), so no need for separate mappings
        # Store (start, length) tuples instead of lists for memory efficiency
        self.doc_id_to_embedding_range = {}  # doc_id -> (start_position, length) tuple
        self.position_to_doc_id = None  # Direct mapping: position -> document ID (numpy array for vectorized indexing)
        self.flattened_embeddings = None  # Flattened embeddings array (only if store_embeddings=True)
        self._documents_added = False  # Track if documents have been added
        
        # Load existing index if index_folder is provided, override is False, and index exists
        if self.index_folder is not None and not self.override:
            index_path = self._get_index_path()
            if index_path is not None:
                scann_config_path = index_path / "scann_config.pb"
                metadata_path = index_path / "metadata.json"
                if scann_config_path.exists() and metadata_path.exists():
                    self._load_index()

    @staticmethod
    def _normalize_verbose(verbose: bool | str) -> str:
        """Normalize `verbose` input to internal string levels."""
        if isinstance(verbose, bool):
            return "init" if verbose else "none"
        return verbose

    def _log_retrieve(self) -> bool:
        return self.verbose_level == "all"

    def _build_searcher(self, embeddings: np.ndarray) -> None:
        """Build the ScaNN searcher from embeddings (in-memory only)."""
        build_start = time.time()
        try:
            import scann
        except ImportError:
            raise ImportError(
                'ScaNN is not installed. Please install it with: `pip install "pylate[scann]"` or `pip install scann`.'
            )

        # Auto-tune parameters if not set (only if not using autopilot)
        num_vectors = embeddings.shape[0]
        self.num_neighbors = self.num_neighbors if self.num_neighbors is not None else min(10, num_vectors)
        
        if self.use_autopilot:
            # When using autopilot, it will auto-tune all parameters
            if self.verbose:
                logger.info(f"[ScaNN] Building ScaNN searcher with {embeddings.shape[0]} vectors using autopilot()...")
                logger.info(f"[ScaNN]   NOTE: autopilot() overrides manual configuration (num_leaves, num_leaves_to_search, training_sample_size)")
                if self.num_leaves is not None or self.num_leaves_to_search is not None or self.training_sample_size is not None:
                    logger.warning(f"[ScaNN]   WARNING: Manual parameters provided but will be ignored: num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search}, training_sample_size={self.training_sample_size}")
        else:
            # Auto-tune parameters if not set
            self.num_leaves = self.num_leaves if self.num_leaves is not None else min(2_000, num_vectors)
            self.num_leaves_to_search = self.num_leaves_to_search if self.num_leaves_to_search is not None else 200
            self.training_sample_size = self.training_sample_size if self.training_sample_size is not None else min(250000, num_vectors)

            if self.verbose:
                logger.info(f"[ScaNN] Building ScaNN searcher with {embeddings.shape[0]} vectors...")
                logger.info(f"[ScaNN]   Parameters: num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search}, training_sample_size={self.training_sample_size}, num_neighbors={self.num_neighbors}")

        # Build ScaNN searcher
        log_memory("Before scann.build()", self.verbose)
        step_start = time.time()
        if self.use_autopilot:
            searcher = (
                scann.scann_ops_pybind.builder(embeddings, self.num_neighbors, "dot_product")
                .autopilot()
                .build()
            )
        else:
            searcher = (
                scann.scann_ops_pybind.builder(embeddings, self.num_neighbors, "dot_product")
                .tree(num_leaves=self.num_leaves, num_leaves_to_search=self.num_leaves_to_search, training_sample_size=self.training_sample_size, spherical=True)
                .score_ah(dimensions_per_block=self.dimensions_per_block, anisotropic_quantization_threshold=self.anisotropic_quantization_threshold)
                .build()
            )
        step_time = time.time() - step_start
        log_memory("After scann.build()", self.verbose)
        if self.verbose:
            logger.info(f"[ScaNN] ScaNN searcher built: {step_time:.4f}s")

        self.searcher = searcher
        
        total_time = time.time() - build_start
        if self.verbose:
            logger.info(f"[ScaNN] Total searcher build time: {total_time:.4f}s")
    
    def _get_index_path(self) -> Path | None:
        """Get the path where the index should be saved/loaded."""
        if self.index_folder is None or self.name is None:
            return None
        index_path = Path(self.index_folder) / self.name
        return index_path
    
    def _load_index(self) -> None:
        """Load an existing index from disk. Raises an error if loading fails."""
        index_path = self._get_index_path()
        if index_path is None:
            raise ValueError(
                f"Cannot load index: index_folder or name not set. "
                f"index_folder={self.index_folder}, name={self.name}"
            )
        
        metadata_path = index_path / "metadata.json"
        doc_id_mapping_path = index_path / "doc_id_to_embedding_range.tsv"
        flattened_embeddings_path = index_path / "flattened_embeddings.npy"
        
        try:
            import scann
        except ImportError:
            raise ImportError(
                'ScaNN is not installed. Cannot load index. '
                'Please install it with: `pip install "pylate[scann]"` or `pip install scann`.'
            )
        
        try:
            if self.verbose:
                logger.info(f"[ScaNN] Loading existing index from {index_path}...")
            
            # Load searcher - use absolute path to avoid path resolution issues
            index_path_abs = index_path.resolve()
            self.searcher = scann.scann_ops_pybind.load_searcher(str(index_path_abs))
            
            # Load metadata (JSON)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                # Restore configuration from metadata
                self.embedding_size = metadata.get("embedding_size", self.embedding_size)
                self.num_neighbors = metadata.get("num_neighbors", self.num_neighbors)
                self.num_leaves = metadata.get("num_leaves", self.num_leaves)
                self.num_leaves_to_search = metadata.get("num_leaves_to_search", self.num_leaves_to_search)
                self.training_sample_size = metadata.get("training_sample_size", self.training_sample_size)
                self.use_autopilot = metadata.get("use_autopilot", self.use_autopilot)
                self.store_embeddings = metadata.get("store_embeddings", self.store_embeddings)
            
            # Load doc_id_to_embedding_range (saved as TSV)
            if doc_id_mapping_path.exists():
                self.doc_id_to_embedding_range = {}
                with open(doc_id_mapping_path, "r") as f:
                    for line in f:
                        doc_id, start, length = line.strip().split("\t")
                        self.doc_id_to_embedding_range[doc_id] = (int(start), int(length))
            else:
                raise FileNotFoundError(f"Document ID mapping not found at {doc_id_mapping_path}")
            
            # Reconstruct position_to_doc_id from doc_id_to_embedding_range
            if self.doc_id_to_embedding_range:
                # Calculate total embeddings from the max end position
                max_end = max(start + length for start, length in self.doc_id_to_embedding_range.values())
                self.position_to_doc_id = np.empty(max_end, dtype=object)
                for doc_id, (start, length) in tqdm(self.doc_id_to_embedding_range.items(), desc="Reconstructing position_to_doc_id", disable=not self.verbose):
                    self.position_to_doc_id[start:start + length] = doc_id
            else:
                self.position_to_doc_id = np.empty(0, dtype=object)
            
            # Load flattened_embeddings if it exists (only if store_embeddings=True)
            if self.store_embeddings and flattened_embeddings_path.exists():
                if self.verbose:
                    logger.info("[ScaNN] Loading flattened_embeddings from %s...", flattened_embeddings_path)
                self.flattened_embeddings = np.load(flattened_embeddings_path)
                if self.verbose:
                    logger.info("[ScaNN] Loaded flattened_embeddings with shape %s", self.flattened_embeddings.shape)
            else:
                self.flattened_embeddings = None
            
            self._documents_added = True
            
            if self.verbose:
                logger.info(f"[ScaNN] Successfully loaded index from {index_path}")
                logger.info(f"[ScaNN]   Documents: {len(self.doc_id_to_embedding_range)}")
                logger.info(f"[ScaNN]   Total embeddings: {len(self.position_to_doc_id) if self.position_to_doc_id is not None else 0}")
        except ImportError:
            # Preserve import errors (e.g. optional dependencies) as-is.
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ScaNN index from {index_path}: {e}. "
                f"This may indicate a corrupted index or version mismatch. "
                f"Set override=True to rebuild the index."
            ) from e
    
    def save(self) -> None:
        """Save the index to disk."""
        if self.searcher is None:
            raise ValueError("Cannot save index: no searcher has been built. Add documents first.")
        
        index_path = self._get_index_path()
        if index_path is None:
            if self.verbose:
                logger.warning("[ScaNN] Cannot save index: index_folder or name not set")
            return
        
        # Create directory if it doesn't exist
        index_path.mkdir(parents=True, exist_ok=True)
        
        metadata_path = index_path / "metadata.json"
        doc_id_mapping_path = index_path / "doc_id_to_embedding_range.tsv"
        flattened_embeddings_path = index_path / "flattened_embeddings.npy"
        
        try:
            if self.verbose:
                logger.info(f"[ScaNN] Saving index to {index_path}...")
            
            # Save searcher - serialize() expects a directory path and will create files inside it
            # Use absolute path to avoid path resolution issues when loading
            # Serialize directly to index_path (not a subdirectory) to avoid path issues
            index_path_abs = index_path.resolve()
            self.searcher.serialize(str(index_path_abs))
            
            # Save metadata as JSON (only simple, serializable values)
            metadata = {
                "embedding_size": self.embedding_size,
                "num_neighbors": self.num_neighbors,
                "num_leaves": self.num_leaves,
                "num_leaves_to_search": self.num_leaves_to_search,
                "training_sample_size": self.training_sample_size,
                "use_autopilot": self.use_autopilot,
                "store_embeddings": self.store_embeddings,
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save doc_id_to_embedding_range as TSV (simple text format)
            # position_to_doc_id can be reconstructed from this, so we don't save it separately
            with open(doc_id_mapping_path, "w") as f:
                for doc_id, (start, length) in tqdm(self.doc_id_to_embedding_range.items(), desc="Saving doc_id_to_embedding_range", disable=not self.verbose):
                    f.write(f"{doc_id}\t{start}\t{length}\n")
            
            # Save flattened_embeddings if store_embeddings=True
            if self.store_embeddings and self.flattened_embeddings is not None:
                np.save(flattened_embeddings_path, self.flattened_embeddings)
            
            if self.verbose:
                logger.info(f"[ScaNN] Index saved successfully to {index_path}")
        except Exception as e:
            logger.error(f"[ScaNN] Failed to save index to {index_path}: {e}")
            raise

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor],
        batch_size: int,
    ) -> "ScaNN":
        """Add documents to the index.
        
        Note: This method only supports adding all documents at once. 
        Subsequent calls will raise an error.
        batch_size is kept for API compatibility but not used.
        """
        # Enforce single add - check if documents already exist
        if self._documents_added:
            raise ValueError(
                "ScaNN index only supports adding all documents at once. "
                "Documents have already been added."
            )
        
        add_start = time.time()
        if self.verbose:
            logger.info(f"[ScaNN] Adding {len(documents_ids)} documents to index...")
        
        log_memory("Start of add_documents", self.verbose)

        # Calculate total embeddings to pre-allocate array
        # Assumes input is list of torch tensors (the standard pylate format)
        step_start = time.time()

        # Get doc lengths and total count in one pass
        doc_lengths = [emb.shape[0] for emb in documents_embeddings]
        total_embeddings = sum(doc_lengths)
        embedding_dim = documents_embeddings[0].shape[1]

        # Preserve incoming dtype (fp16/fp32) to avoid holding two full flattened
        # copies (e.g., one fp16 and one fp32) in memory at the same time.
        input_dtype = documents_embeddings[0].dtype
        if input_dtype not in (torch.float16, torch.float32):
            raise ValueError(
                "ScaNN expects document embeddings to be float16 or float32. "
                f"Got dtype={input_dtype}."
            )
        np_dtype = np.float16 if input_dtype == torch.float16 else np.float32
        
        if self.verbose:
            size_gb = total_embeddings * embedding_dim * np.dtype(np_dtype).itemsize / 1e9
            logger.info(
                f"[ScaNN] Pre-allocating array for {total_embeddings} embeddings x "
                f"{embedding_dim} dims ({size_gb:.2f} GB) with dtype={np_dtype}"
            )
        
        # Pre-allocate flattened array in the same dtype as incoming embeddings.
        flattened_embeddings = np.empty((total_embeddings, embedding_dim), dtype=np_dtype)
        
        log_memory("After pre-allocating flattened_embeddings array", self.verbose)
        
        # Fill array in-place, deleting each tensor after copying to free memory
        offset = 0
        num_docs = len(documents_embeddings)
        log_interval = max(1, num_docs // 10)  # Log memory ~10 times during the loop
        
        iterator = tqdm(
            enumerate(documents_embeddings),
            desc="Flattening documents and adding to pre-allocated array",
            total=num_docs,
            disable=not self.verbose,
        )
        for i, emb in iterator:
            if emb.dtype != input_dtype:
                raise ValueError(
                    "All document embeddings must have the same dtype. "
                    f"Expected {input_dtype}, got {emb.dtype}."
                )
            n = emb.shape[0]
            flattened_embeddings[offset:offset + n] = emb.to(
                "cpu", dtype=input_dtype
            ).numpy()
            offset += n
            
            # Log memory periodically
            if self.verbose and (i + 1) % log_interval == 0:
                log_memory(f"During fill loop ({i + 1}/{num_docs} docs, {offset}/{total_embeddings} embeddings)", self.verbose)
        
        log_memory("After fill loop, before gc", self.verbose)
        
        # Clear the list and run gc
        del documents_embeddings
        gc.collect()
        
        log_memory("After del documents_embeddings + gc.collect()", self.verbose)
        
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(
                f"[ScaNN] Flattened {total_embeddings} embeddings to {np_dtype}: "
                f"{step_time:.4f}s"
            )
        
        # Build position->doc_id array and doc_id->embedding_range mapping
        step_start = time.time()
        self.position_to_doc_id = np.empty(total_embeddings, dtype=object)
        offset = 0
        for doc_id, num_tokens in zip(documents_ids, doc_lengths):
            # Store (start, length) tuple instead of list for memory efficiency
            self.doc_id_to_embedding_range[doc_id] = (offset, num_tokens)
            # Broadcast doc_id to fill the slice (no temp list needed)
            self.position_to_doc_id[offset:offset + num_tokens] = doc_id
            offset += num_tokens
        
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Built ID mappings and position->doc_id array: {step_time:.4f}s")

        # Build the ScaNN index with all embeddings
        if len(flattened_embeddings) > 0:
            if self.verbose:
                logger.info(f"[ScaNN] Building index with {len(flattened_embeddings)} embeddings...")
            
            log_memory("Before _build_searcher", self.verbose)
            
            # Note: embedding_id == position (sequential), so no position mappings needed
            # Build searcher (in-memory only)
            self._build_searcher(flattened_embeddings)
            
            # Store flattened embeddings if requested, otherwise free the array
            if self.store_embeddings:
                self.flattened_embeddings = flattened_embeddings
                log_memory("After _build_searcher + storing flattened_embeddings reference", self.verbose)
            else:
                del flattened_embeddings
                gc.collect()
                log_memory("After _build_searcher + del flattened_embeddings", self.verbose)
            
            # Mark that documents have been added
            self._documents_added = True
            
            # Save index to disk if index_folder is set
            if self.index_folder is not None:
                self.save()
            
            total_time = time.time() - add_start
            if self.verbose:
                logger.info(f"[ScaNN] Total add_documents time: {total_time:.4f}s")

        return self

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Remove documents from the index.
        
        Not supported for ScaNN index.

        Parameters
        ----------
        documents_ids
            The documents IDs to remove.

        Raises
        ------
        NotImplementedError
            Document removal is not supported for ScaNN index.

        """
        raise NotImplementedError(
            "Document removal is not supported for ScaNN index."
        )

    def __call__(
        self,
        queries_embeddings: list[list[int | float]],
        k: int = 5,
        subset: list[list[str]] | list[str] | None = None,
    ) -> dict:
        """Query the index for the nearest neighbors of the queries embeddings.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of nearest neighbors to return.
        subset
            Optional subset of document IDs to restrict search to.
            Not yet implemented for ScaNN index.

        Raises
        ------
        NotImplementedError
            If subset is provided (not yet implemented).

        """
        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not yet implemented for ScaNN index."
            )
        
        if self.searcher is None:
            raise ValueError("Index is empty, add documents before querying.")

        total_start = time.time()
        
        # Reshape queries
        step_start = time.time()
        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
        n_queries = len(queries_embeddings)
        step_time = time.time() - step_start
        if self._log_retrieve():
            logger.info(f"[ScaNN] Reshaping {n_queries} queries: {step_time:.4f}s")

        # Flatten query embeddings (assume they are already normalized)
        step_start = time.time()
        flattened_queries = np.array(
            list(itertools.chain(*queries_embeddings))
        )
        n_tokens_total = len(flattened_queries)
        step_time = time.time() - step_start
        if self._log_retrieve():
            logger.info(f"[ScaNN] Flattening {n_tokens_total} query tokens: {step_time:.4f}s")

        # Query the index
        step_start = time.time()
        neighbors, distances = self.searcher.search_batched_parallel(flattened_queries, final_num_neighbors=k)
        # replace NaN values with 0
        if np.isnan(distances).any():
            logger.warning("[ScaNN] distances has %d NaN values out of %d total; replacing with 0", np.isnan(distances).sum(), distances.size)
            distances = np.nan_to_num(distances, nan=0.0)
        step_time = time.time() - step_start
        if self._log_retrieve():
            logger.info(f"[ScaNN] ScaNN search_batched for {n_tokens_total} tokens (k={k}): {step_time:.4f}s ({step_time/n_tokens_total*1000:.2f}ms per token)")

        # Map embedding indices back to document IDs using fully vectorized numpy operations
        step_start = time.time()
        n_tokens_per_query = [len(q) for q in queries_embeddings]
        
        # Vectorized lookup: process all tokens at once using numpy advanced indexing
        # neighbors shape: (n_tokens_total, k), distances shape: (n_tokens_total, k)
        all_doc_ids = self.position_to_doc_id[neighbors]  # Vectorized lookup for all tokens
        all_distances = distances
        
        # Reshape back into nested structure (queries -> tokens -> neighbors)
        documents = []
        distances_list = []
        token_idx = 0
        for n_tokens in n_tokens_per_query:
            query_documents = []
            query_distances = []
            
            for _ in range(n_tokens):
                # Extract results for this token (already vectorized)
                token_docs = all_doc_ids[token_idx]
                token_dists = all_distances[token_idx]
                
                query_documents.append(token_docs)
                query_distances.append(token_dists)
                token_idx += 1
            
            documents.append(query_documents)
            distances_list.append(query_distances)

        step_time = time.time() - step_start
        if self._log_retrieve():
            logger.info(f"[ScaNN] Mapping results to document IDs: {step_time:.4f}s")
        
        total_time = time.time() - total_start
        if self._log_retrieve():
            logger.info(f"[ScaNN] Total retrieval time: {total_time:.4f}s ({total_time/n_queries*1000:.2f}ms per query, {total_time/n_tokens_total*1000:.2f}ms per token)")

        return {
            "documents_ids": documents,
            "distances": distances_list,  # Keep as list to handle variable-length query tokens (ragged)
        }

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[np.ndarray]]:
        """Get document embeddings by their IDs.
        
        Parameters
        ----------
        documents_ids
            Nested list of document IDs. Each inner list represents a group of documents.
        
        Returns
        -------
        list[list[np.ndarray]]
            Nested list of embeddings. Each embedding is a numpy array with shape (seq_len, dim).
        
        Raises
        ------
        NotImplementedError
            If store_embeddings=False (embeddings are not stored).
        ValueError
            If index is empty or document ID not found.
        """
        if not self.store_embeddings:
            raise NotImplementedError(
                "Retrieving document embeddings requires store_embeddings=True. "
                "Set store_embeddings=True when creating the index."
            )
        
        if self.flattened_embeddings is None:
            raise ValueError("Index is empty, add documents before retrieving embeddings.")
        
        reconstructed_embeddings = []
        for doc_group in documents_ids:
            group_embeddings = []
            for doc_id in doc_group:
                if doc_id not in self.doc_id_to_embedding_range:
                    raise ValueError(f"Document ID '{doc_id}' not found in index.")
                
                start, length = self.doc_id_to_embedding_range[doc_id]
                # Slice the flattened array to get document embeddings
                doc_emb = self.flattened_embeddings[start:start + length]
                group_embeddings.append(doc_emb)
            reconstructed_embeddings.append(group_embeddings)
        
        return reconstructed_embeddings
