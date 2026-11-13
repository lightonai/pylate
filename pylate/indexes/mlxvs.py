from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .base import Base
from .utils import np_dtype_for, reshape_embeddings

logger = logging.getLogger(__name__)


class MLXVS(Base):
    """MLXVS index: token-level nearest neighbor search on the GPU of Apple silicon Macs, with
    [mlxvs](https://github.com/mikepapadim/mlxvs) (built on Apple's MLX).

    **Important Notes:**
    - Runs on Apple silicon Macs only (macOS 14 or later)
    - By default the search is **exact**: every query token is scored against every document token on the GPU
    - With ``exact=False`` it searches a CAGRA graph index (the graph search of NVIDIA cuVS), which is faster on
      large collections and extended in place when documents are added
    - Documents can be added in several calls and removed

    To use this index, you need to install the `mlxvs` extra:

    ```bash
    pip install "pylate[mlxvs]"
    ```

    or install mlxvs directly:

    ```bash
    pip install mlxvs
    ```

    Parameters
    ----------
    index_name
        The name of the index collection.
    embedding_size
        The number of dimensions of the embeddings.
    exact
        Whether to score every document token (True) or search a CAGRA graph index (False).
    graph_degree
        The number of neighbors of each token in the graph index (``exact=False``).
    intermediate_graph_degree
        The number of neighbors of the k-NN graph pruned into the graph index (``exact=False``).
    itopk_size
        The length of each query token's search list in the graph index (``exact=False``): longer is more
        accurate and slower. At least the number of neighbors asked for.
    index_folder
        The folder where the index is saved and loaded. If None, the index is not persisted to disk.
    override
        Whether to override the index if it already exists. If False and the index exists, it is loaded.
    verbose
        Whether to log the index's build and load steps.

    """

    is_end_to_end_index = False

    def __init__(
        self,
        index_name: str | None = "MLXVS_index",
        embedding_size: int = 128,
        exact: bool = True,
        graph_degree: int = 32,
        intermediate_graph_degree: int = 64,
        itopk_size: int = 64,
        index_folder: str | None = None,
        override: bool = False,
        verbose: bool = False,
    ) -> None:
        try:
            import mlx.core as mx
            import mlxvs
        except ImportError as error:
            raise ImportError(
                "The MLXVS index needs mlxvs, which runs on Apple silicon Macs: `pip install mlxvs`."
            ) from error
        self._mx = mx
        self._mlxvs = mlxvs

        self.index_name = index_name
        self.embedding_size = embedding_size
        self.exact = exact
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree
        self.itopk_size = itopk_size
        self.index_folder = index_folder
        self.override = override
        self.verbose = verbose

        # Every document token's embedding, in insertion order (numpy, fp16 or fp32), and the document of each
        self.embeddings = None
        self.position_to_doc_id = np.empty(0, dtype=object)
        # doc_id -> (start position, number of tokens); removed documents are dropped from it
        self.doc_id_to_embedding_range = {}
        # Tokens of removed documents (searched but never returned by the graph index)
        self.removed = np.zeros(0, dtype=bool)
        # The tokens on the GPU (exact search) or their graph index
        self._rows = None
        self._graph = None

        index_path = self._get_index_path()
        if (
            index_path is not None
            and not self.override
            and (index_path / "metadata.json").exists()
        ):
            self._load_index()

    def _get_index_path(self) -> Path | None:
        if self.index_folder is None:
            return None
        return Path(self.index_folder) / self.index_name

    @staticmethod
    def _to_numpy(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
        if not isinstance(embeddings, (torch.Tensor, np.ndarray)):
            raise ValueError(
                "MLXVS expects document embeddings to be torch.Tensor or np.ndarray. "
                f"Got type={type(embeddings)}."
            )
        if np_dtype_for(embeddings.dtype) is None:
            raise ValueError(
                f"MLXVS expects float16 or float32 embeddings. Got dtype={embeddings.dtype}."
            )
        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().to("cpu").numpy()
        return embeddings

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor | np.ndarray],
        batch_size: int = 128,
    ) -> "MLXVS":
        """Add documents to the index. Documents whose ID is already in the index replace it.

        batch_size is kept for API compatibility but not used.
        """
        if len(documents_ids) != len(documents_embeddings):
            raise ValueError(
                "documents_ids and documents_embeddings must have the same length."
            )
        if len(documents_embeddings) == 0:
            return self

        existing = [
            doc_id
            for doc_id in documents_ids
            if doc_id in self.doc_id_to_embedding_range
        ]
        if existing:
            self.remove_documents(existing)

        new = [self._to_numpy(embeddings) for embeddings in documents_embeddings]
        dtypes = {embeddings.dtype for embeddings in new}
        if self.embeddings is not None:
            dtypes.add(self.embeddings.dtype)
        if len(dtypes) > 1:
            raise ValueError(
                f"All document embeddings must have the same dtype. Got {sorted(map(str, dtypes))}."
            )
        dims = {embeddings.shape[1] for embeddings in new}
        if self.embeddings is not None:
            dims.add(self.embeddings.shape[1])
        if len(dims) > 1:
            raise ValueError(
                "All document embeddings must have the same embedding dimension."
            )

        start = 0 if self.embeddings is None else len(self.embeddings)
        flattened = np.concatenate(new, axis=0)
        position_to_doc_id = np.empty(len(flattened), dtype=object)
        offset = 0
        for doc_id, embeddings in zip(documents_ids, new):
            self.doc_id_to_embedding_range[doc_id] = (
                start + offset,
                embeddings.shape[0],
            )
            position_to_doc_id[offset : offset + embeddings.shape[0]] = doc_id
            offset += embeddings.shape[0]

        if self.verbose:
            logger.info(
                f"[MLXVS] Adding {len(documents_ids)} documents ({len(flattened)} tokens) to the index."
            )

        self.embeddings = (
            flattened
            if self.embeddings is None
            else np.concatenate([self.embeddings, flattened])
        )
        self.position_to_doc_id = np.concatenate(
            [self.position_to_doc_id, position_to_doc_id]
        )
        self.removed = np.concatenate(
            [self.removed, np.zeros(len(flattened), dtype=bool)]
        )
        self._add_to_search(flattened)

        if self.index_folder is not None:
            self.save()
        return self

    def _add_to_search(self, flattened: np.ndarray) -> None:
        mx = self._mx
        if self.exact:
            self._rows = mx.array(self.embeddings)
            return
        if (
            self._graph is None
            and len(self.embeddings) > self.intermediate_graph_degree
        ):
            if self.verbose:
                logger.info(
                    f"[MLXVS] Building the graph index of {len(self.embeddings)} tokens."
                )
            self._graph = self._mlxvs.CagraIndex.build(
                self.embeddings,
                graph_degree=self.graph_degree,
                metric="inner_product",
                intermediate_graph_degree=self.intermediate_graph_degree,
                dtype="float16" if self.embeddings.dtype == np.float16 else "float32",
            )
        elif self._graph is not None:
            self._graph.extend(flattened)
        # Collections too small for a graph are searched exactly
        self._rows = mx.array(self.embeddings) if self._graph is None else None

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Remove documents from the index. Their tokens are no longer returned by searches."""
        for doc_id in documents_ids:
            if doc_id not in self.doc_id_to_embedding_range:
                continue
            start, length = self.doc_id_to_embedding_range.pop(doc_id)
            self.removed[start : start + length] = True
        if (self.exact or self._graph is None) and self.removed.any():
            # exact search keeps only the remaining tokens on the GPU
            self._compact()
        if self.index_folder is not None:
            self.save()

    def _compact(self) -> None:
        keep = ~self.removed
        self.embeddings = self.embeddings[keep]
        self.position_to_doc_id = self.position_to_doc_id[keep]
        self.removed = np.zeros(len(self.embeddings), dtype=bool)
        offset = 0
        ranges = {}
        for doc_id in dict.fromkeys(self.position_to_doc_id.tolist()):
            length = self.doc_id_to_embedding_range[doc_id][1]
            ranges[doc_id] = (offset, length)
            offset += length
        self.doc_id_to_embedding_range = ranges
        self._rows = self._mx.array(self.embeddings) if len(self.embeddings) else None

    def __call__(
        self,
        queries_embeddings: list[list[int | float]],
        k: int = 10,
    ) -> dict:
        """Query the index for the nearest neighbors of the queries embeddings.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of nearest neighbors to return per query token.

        Returns
        -------
        dict
            ``documents_ids`` and ``distances`` (cosine distances, ``1 - inner product`` of the normalized
            embeddings), per query and query token.

        """
        mx = self._mx
        if self.embeddings is None or len(self.doc_id_to_embedding_range) == 0:
            raise ValueError("Index is empty, add documents before querying.")

        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
        n_tokens_per_query = [len(query) for query in queries_embeddings]
        flattened_queries = np.asarray(
            list(itertools.chain(*queries_embeddings)), dtype=np.float32
        )
        queries = mx.array(flattened_queries)

        if self._graph is not None:
            k = min(k, int((~self.removed).sum()))
            options = {"itopk_size": max(self.itopk_size, k)}
            if self.removed.any():
                options["filter"] = ~self.removed
            ids, scores = self._graph.search(queries, k, **options)
        else:
            k = min(k, len(self.embeddings))
            if k <= 256:
                ids, scores = self._mlxvs.brute_force_knn(
                    self._rows, queries, k, "inner_product"
                )
            else:
                all_scores = queries @ self._rows.astype(mx.float32).T
                ids = mx.argsort(-all_scores, axis=1)[:, :k]
                scores = mx.take_along_axis(all_scores, ids, axis=1)
        neighbors = np.array(ids, dtype=np.int64)
        distances = 1.0 - np.array(scores, dtype=np.float32)

        all_doc_ids = self.position_to_doc_id[np.maximum(neighbors, 0)]
        splits = np.cumsum(n_tokens_per_query[:-1])
        documents = [list(chunk) for chunk in np.split(all_doc_ids, splits)]
        distances_list = [list(chunk) for chunk in np.split(distances, splits)]
        return {"documents_ids": documents, "distances": distances_list}

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[np.ndarray]]:
        """Get document embeddings by their IDs, as a nested list of ``(n_tokens, dim)`` arrays."""
        reconstructed_embeddings = []
        for doc_group in documents_ids:
            group_embeddings = []
            for doc_id in doc_group:
                if doc_id not in self.doc_id_to_embedding_range:
                    raise ValueError(f"Document ID '{doc_id}' not found in index.")
                start, length = self.doc_id_to_embedding_range[doc_id]
                group_embeddings.append(self.embeddings[start : start + length])
            reconstructed_embeddings.append(group_embeddings)
        return reconstructed_embeddings

    def save(self) -> None:
        """Save the index to ``index_folder/index_name``."""
        index_path = self._get_index_path()
        if index_path is None:
            raise ValueError("index_folder must be set to save the index.")
        index_path.mkdir(parents=True, exist_ok=True)
        np.save(
            index_path / "embeddings.npy",
            np.zeros((0, self.embedding_size))
            if self.embeddings is None
            else self.embeddings,
        )
        np.save(index_path / "removed.npy", self.removed)
        metadata = {
            "exact": self.exact,
            "documents_ids": self.position_to_doc_id.tolist(),
            "ranges": {
                doc_id: list(r) for doc_id, r in self.doc_id_to_embedding_range.items()
            },
        }
        with open(index_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        if self._graph is not None:
            self._graph.save(str(index_path / "graph.safetensors"))

    def _load_index(self) -> None:
        index_path = self._get_index_path()
        if self.verbose:
            logger.info(f"[MLXVS] Loading the index from {index_path}.")
        with open(index_path / "metadata.json") as f:
            metadata = json.load(f)
        self.exact = metadata["exact"]
        self.embeddings = np.load(index_path / "embeddings.npy")
        self.removed = np.load(index_path / "removed.npy")
        self.position_to_doc_id = np.array(metadata["documents_ids"], dtype=object)
        self.doc_id_to_embedding_range = {
            doc_id: tuple(r) for doc_id, r in metadata["ranges"].items()
        }
        graph_path = index_path / "graph.safetensors"
        if not self.exact and graph_path.exists():
            self._graph = self._mlxvs.CagraIndex.load(str(graph_path))
        elif len(self.embeddings):
            self._rows = self._mx.array(self.embeddings)
