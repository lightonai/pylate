from __future__ import annotations

import logging
import os
import pickle
import shutil
import warnings

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)

_SENTINEL_DOC_ID = np.iinfo(np.uint32).max


class TachiomIndex(Base):
    """TACHIOM index for late-interaction multi-vector retrieval.

    Wraps the Rust-backed TACHIOM library (TAC + PQ + HNSW) as a drop-in PyLate
    index. Token-Aware Clustering groups token embeddings by vocabulary ID before
    k-means, which improves clustering speed and retrieval quality over standard k-means.

    Encode documents with ``output_value=None`` to enable Token-Aware Clustering.
    The returned dict (keys: ``"token_embeddings"``, ``"input_ids"``, ``"masks"``,
    ``"attention_mask"``) carries vocabulary token IDs that ``add_documents``
    extracts automatically:

        embeddings = model.encode(docs, is_query=False, output_value=None)
        index.add_documents(doc_ids, embeddings)

    If ``documents_token_ids`` is omitted, all tokens are assigned ID 0 so TAC
    degrades to a single global k-means. A ``UserWarning`` is issued.

    Parameters
    ----------
    index_folder
        Directory that will contain the index sub-folder.
    index_name
        Name of the index sub-folder inside ``index_folder``.
    override
        Delete and recreate the index directory if it already exists.
    center_dataset
        Subtract the global mean token vector from all document vectors before
        building the index. Default: ``True``. May improve HNSW quality.
    total_centroids
        TAC coarse-centroid budget. ``None`` (default) auto-computes as
        ``max(2^round(log2(n_tokens/128)), ceil(min_tac_budget * 1.1))``,
        ensuring TAC is used rather than falling back to global k-means.
    tac_n_iter
        K-means iterations for Token-Aware Clustering. Default: 10. Reduce for
        fast experimentation; raise for maximum quality. 10 is usually enough.
    tac_micro_threshold
        Token groups with fewer vectors than this receive 1 centroid each.
        ``None`` (default) auto-derives as
        ``2^round(log2(n_tokens^0.25))`` clamped to ``[32, 128]``.
    tac_small_threshold
        Token groups in ``[micro, small)`` receive 2 centroids each.
        ``None`` (default) auto-derives as ``2 * tac_micro_threshold``.
    pq_sample_size
        Maximum number of vectors sampled for PQ codebook training. Default:
        10,000,000. Safe to lower on small corpora. May be
        increased on very large datasets (> 1B tokens).
    pq_n_iter
        K-means iterations for PQ codebook training. Default: 10. Same
        trade-off as ``tac_n_iter``.
    normalize
        L2-normalise residuals before PQ encoding. Default: ``True``. Leave
        as ``True`` unless you have a specific reason to disable.
    pq_seed
        Random seed for PQ codebook training. Default: 42. Fix for
        reproducibility; change to get a different codebook.
    hnsw_m
        HNSW graph degree (edges per node). Default: 32. Higher = better
        recall and more memory. Typical range: 16–64.
    ef_construction
        HNSW build-time search width. Default: 1500. Purposefully high to
        maximise recall; up to 1–2M centroids the HNSW build time is
        negligible compared to TAC and PQ. For very large centroid counts
        (> 2M) reduce to keep build times reasonable. Hardly gives benefits
        above 1500.
    k_centroids
        Coarse centroids probed per query token at search time (``n_probe`` in
        most IVF-based algorithms). Default: 20. Higher = more candidates,
        better recall, slower search.
    k_docs_to_score
        Candidate pool size for full late-interaction MaxSim scoring. Default:
        500. Must be ≥ ``k``. Alpha pruning may further reduce this pool
        before MaxSim. Increase for higher recall at the cost of latency.
    ef_search
        HNSW search-time exploration width. ``None`` (default) resolves to
        ``round(1.5 × k_centroids)`` at search time, keeping the two coupled
        automatically. Set explicitly only to deviate from the 1.5× rule.
    alpha
        Coarse-score pruning threshold: candidates whose coarse score falls
        more than ``alpha × score[k]`` below the k-th best are dropped before
        MaxSim. Default: 0.45. Range [0, 1]; usually effective in [0, 0.5].
        Smaller = more aggressive pruning = faster search but worse recall.
        ``None`` disables pruning (all ``k_docs_to_score`` candidates scored).
    beta
        Early-termination patience: stop MaxSim scoring after this many
        consecutive non-improving documents. ``None`` = disabled (score all).
    lambda_
        HNSW early-exit parameter. Makes search faster; tune together with
        ``ef_search``. ``None`` = disabled.
    num_threads
        Worker threads for ``batch_search``. 0 = rayon default (all cores),
        1 = single-threaded, n = custom pool of size n.

    References
    ----------
    - [Martinico et al., "Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing", SIGIR 2026](https://arxiv.org/abs/2604.28142)
    - [TACHIOM GitHub repository](https://github.com/TusKANNy/tachiom)

    If you use TACHIOM in your research, please cite::

        @misc{martinico2026efficientmultivectorretrievaltokenaware,
              title={Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing},
              author={Silvio Martinico and Franco Maria Nardini and Cosimo Rulli and Rossano Venturini},
              year={2026},
              eprint={2604.28142},
              archivePrefix={arXiv},
              primaryClass={cs.IR},
              url={https://arxiv.org/abs/2604.28142},
        }
    """

    is_end_to_end_index = True

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "tachiom",
        override: bool = False,
        # Build hyperparams
        center_dataset: bool = True,
        total_centroids: int | None = None,
        tac_n_iter: int = 10,
        tac_micro_threshold: int | None = None,
        tac_small_threshold: int | None = None,
        pq_sample_size: int = 10_000_000,
        pq_n_iter: int = 10,
        normalize: bool = True,
        pq_seed: int = 42,
        hnsw_m: int = 32,
        ef_construction: int = 1500,
        # Search hyperparams
        k_centroids: int = 20,
        k_docs_to_score: int = 500,
        ef_search: int | None = None,
        alpha: float | None = 0.45,
        beta: int | None = None,
        lambda_: float | None = None,
        num_threads: int = 0,
    ) -> None:
        try:
            from tachiom import Tachiom as _Tachiom
            from tachiom import auto_build_params as _auto_build_params
        except ImportError:
            raise ImportError(
                "tachiom is not installed. Please install it with: `pip install tachiom`."
            )
        self._Tachiom = _Tachiom
        self._auto_build_params = _auto_build_params

        self.index_folder = index_folder
        self.index_name = index_name

        self.center_dataset = center_dataset
        self.total_centroids = total_centroids
        self.tac_n_iter = tac_n_iter
        self.tac_micro_threshold = tac_micro_threshold
        self.tac_small_threshold = tac_small_threshold
        self.pq_sample_size = pq_sample_size
        self.pq_n_iter = pq_n_iter
        self.normalize = normalize
        self.pq_seed = pq_seed
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction

        self.k_centroids = k_centroids
        self.k_docs_to_score = k_docs_to_score
        self.ef_search = ef_search
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.num_threads = num_threads

        self.index_path = os.path.join(index_folder, index_name)
        if override and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        self._tachiom_path = os.path.join(self.index_path, "tachiom.bin")
        self._doc_id_to_int_path = os.path.join(self.index_path, "doc_id_to_int.pkl")
        self._int_to_doc_id_path = os.path.join(self.index_path, "int_to_doc_id.pkl")

        self._index = None
        self._doc_id_to_int: dict | None = None
        self._int_to_doc_id: dict | None = None
        self.is_indexed = os.path.exists(self._tachiom_path)
        if self.is_indexed:
            self._ensure_loaded()
            self._ensure_mappings()

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = self._Tachiom.load(self._tachiom_path)

    def _ensure_mappings(self) -> None:
        if self._doc_id_to_int is None:
            if not os.path.exists(self._doc_id_to_int_path):
                raise FileNotFoundError(
                    f"Document ID mapping not found at {self._doc_id_to_int_path}. "
                    "Please call add_documents before querying."
                )
            with open(self._doc_id_to_int_path, "rb") as f:
                self._doc_id_to_int = pickle.load(f)
        if self._int_to_doc_id is None:
            if not os.path.exists(self._int_to_doc_id_path):
                raise FileNotFoundError(
                    f"Document ID mapping not found at {self._int_to_doc_id_path}. "
                    "Please call add_documents before querying."
                )
            with open(self._int_to_doc_id_path, "rb") as f:
                self._int_to_doc_id = pickle.load(f)

    def _save_mappings(self, doc_id_to_int: dict, int_to_doc_id: dict) -> None:
        with open(self._doc_id_to_int_path, "wb") as f:
            pickle.dump(doc_id_to_int, f)
        with open(self._int_to_doc_id_path, "wb") as f:
            pickle.dump(int_to_doc_id, f)
        self._doc_id_to_int = doc_id_to_int
        self._int_to_doc_id = int_to_doc_id

    @staticmethod
    def _to_f32(emb: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().numpy()
        return np.asarray(emb, dtype=np.float32)

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        *,
        documents_token_ids: list[np.ndarray] | None = None,
        **kwargs,
    ) -> "TachiomIndex":
        """Index a set of documents.

        Parameters
        ----------
        documents_ids
            String identifiers for the documents.
        documents_embeddings
            Per-document token embeddings. Either:

            * A list of arrays/tensors, each shape ``(n_tokens, dim)`` float32/float16.
            * A dict from ``model.encode(..., output_value=None)`` with keys
              ``"token_embeddings"``, ``"input_ids"``, and ``"masks"``; the mask
              is applied automatically and ``documents_token_ids`` is ignored.
        documents_token_ids
            Vocabulary token IDs aligned with ``documents_embeddings``, each
            of shape ``(n_tokens,)`` in uint32. Obtain from
            ``model.encode(..., output_value=None)`` (preferred) or
            ``model.encode(...)``. Ignored when ``documents_embeddings`` is a dict.
            If ``None`` and a dict is not used, all tokens are assigned ID 0
            (TAC degrades to global k-means) and a ``UserWarning`` is issued.
        kwargs
            Accepted for interface compatibility with other ``Base`` indexes
            (e.g. WARP) and ignored.
        """
        if self.is_indexed:
            warnings.warn(
                "TachiomIndex has already been built and does not currently support "
                "incremental updates. This call has no effect.",
                UserWarning,
                stacklevel=2,
            )
            return self

        if isinstance(documents_embeddings, dict):
            if documents_token_ids is not None:
                warnings.warn(
                    "documents_token_ids is ignored when documents_embeddings is a dict "
                    "(from output_value=None); token IDs are read from the dict.",
                    UserWarning,
                    stacklevel=2,
                )
            embeddings_f32 = []
            documents_token_ids = []
            for emb, mask, ids in zip(
                documents_embeddings["token_embeddings"],
                documents_embeddings["masks"],
                documents_embeddings["input_ids"],
            ):
                embeddings_f32.append(self._to_f32(emb[mask]))
                ids = ids[mask]
                if isinstance(ids, torch.Tensor):
                    documents_token_ids.append(ids.cpu().numpy().astype(np.uint32))
                else:
                    documents_token_ids.append(np.asarray(ids, dtype=np.uint32))
        else:
            embeddings_f32 = [self._to_f32(e) for e in documents_embeddings]

        new_doclens = np.array([e.shape[0] for e in embeddings_f32], dtype=np.int32)

        # Write directly into a pre-allocated float16 buffer instead of
        # np.vstack(...).astype(float16), which would allocate a second
        # full-corpus float32 array before the float16 conversion.
        new_vectors_f16 = np.empty(
            (int(new_doclens.sum()), embeddings_f32[0].shape[1]), dtype=np.float16
        )
        offset = 0
        for e in embeddings_f32:
            new_vectors_f16[offset : offset + e.shape[0]] = e
            offset += e.shape[0]
        new_vectors_u16 = new_vectors_f16.view(np.uint16)

        if documents_token_ids is not None:
            new_token_ids = np.concatenate(
                [np.asarray(t, dtype=np.uint32) for t in documents_token_ids]
            )
        else:
            warnings.warn(
                "documents_token_ids not provided; all tokens will be assigned ID 0, "
                "so TAC degrades to standard k-means over all vectors. "
                "This is significantly slower than TAC and reduces retrieval quality. "
                "Pass documents_token_ids from model.encode(..., output_value=None) "
                "to enable Token-Aware Clustering.",
                UserWarning,
                stacklevel=2,
            )
            new_token_ids = np.zeros(int(new_doclens.sum()), dtype=np.uint32)

        doc_id_to_int = {doc_id: i for i, doc_id in enumerate(documents_ids)}
        int_to_doc_id = {i: doc_id for i, doc_id in enumerate(documents_ids)}

        token_ids_cont = np.ascontiguousarray(new_token_ids, dtype=np.uint32)
        params = self._auto_build_params(
            token_ids_cont,
            total_centroids=self.total_centroids,
            tac_micro_threshold=self.tac_micro_threshold,
            tac_small_threshold=self.tac_small_threshold,
        )
        total_centroids = params["total_centroids"]
        micro_threshold = params["tac_micro_threshold"]
        small_threshold = params["tac_small_threshold"]
        # Store resolved values so __repr__ can show the actual config used.
        self.total_centroids = total_centroids
        self.tac_micro_threshold = micro_threshold
        self.tac_small_threshold = small_threshold
        logger.debug(
            "TachiomIndex: %d docs, %d tokens, dim=%d — "
            "total_centroids=%d, micro_threshold=%d, small_threshold=%d",
            len(new_doclens),
            len(new_token_ids),
            embeddings_f32[0].shape[1],
            total_centroids,
            micro_threshold,
            small_threshold,
        )
        self._index = self._Tachiom.build_from_arrays(
            vectors=new_vectors_u16,
            token_ids=token_ids_cont,
            doclens=new_doclens,
            center_dataset=self.center_dataset,
            total_centroids=total_centroids,
            tac_n_iter=self.tac_n_iter,
            tac_micro_threshold=micro_threshold,
            tac_small_threshold=small_threshold,
            pq_sample_size=self.pq_sample_size,
            pq_n_iter=self.pq_n_iter,
            normalize=self.normalize,
            pq_seed=self.pq_seed,
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
        )
        self._index.save(self._tachiom_path)
        self._save_mappings(doc_id_to_int, int_to_doc_id)
        self.is_indexed = True

        return self

    def __call__(
        self,
        queries_embeddings: (
            np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor]
        ),
        k: int = 10,
    ) -> list[list[RerankResult]]:
        """Search the index for the nearest documents to each query.

        Parameters
        ----------
        queries_embeddings
            Query token embeddings. Accepts a list of 2D arrays
            ``(n_tokens, dim)``, a single 2D array (treated as one query),
            or a 3D array ``(n_queries, n_tokens, dim)``.
        k
            Number of results to return per query.
        """
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before querying."
            )

        self._ensure_loaded()
        self._ensure_mappings()

        # Normalise to list of 2D f32 arrays.
        # PyLate's encode() with padding=True returns (1, Q, dim) tensors per query
        # due to torch.split(..., 1, dim=0) — squeeze that leading 1 away.
        if isinstance(queries_embeddings, (np.ndarray, torch.Tensor)):
            arr = self._to_f32(queries_embeddings)
            if arr.ndim == 2:
                queries_list = [arr]
            else:
                queries_list = [arr[i] for i in range(arr.shape[0])]
        else:
            queries_list = [self._to_f32(q) for q in queries_embeddings]

        queries_list = [
            q.squeeze(0) if q.ndim == 3 and q.shape[0] == 1 else q for q in queries_list
        ]

        n_queries = len(queries_list)
        lens = [q.shape[0] for q in queries_list]
        tokens = np.ascontiguousarray(np.vstack(queries_list), dtype=np.float32)

        if len(set(lens)) == 1:
            offsets = None
        else:
            offsets = np.zeros(n_queries + 1, dtype=np.uint64)
            np.cumsum(lens, out=offsets[1:])

        scores, doc_ids = self._index.batch_search(
            tokens=tokens,
            n_queries=n_queries,
            k=k,
            offsets=offsets,
            num_threads=self.num_threads,
            k_centroids=self.k_centroids,
            k_docs_to_score=self.k_docs_to_score,
            ef_search=self.ef_search,
            alpha=self.alpha,
            beta=self.beta,
            lambda_=self.lambda_,
        )

        results = []
        for query_scores, query_doc_ids in zip(scores, doc_ids):
            query_results = []
            for score, doc_id in zip(query_scores, query_doc_ids):
                if doc_id == _SENTINEL_DOC_ID:
                    break
                query_results.append(
                    RerankResult(
                        id=self._int_to_doc_id[int(doc_id)], score=float(score)
                    )
                )
            results.append(query_results)

        return results

    def remove_documents(self, documents_ids: list[str]) -> None:
        warnings.warn(
            "TachiomIndex does not currently support document removal. This call has no effect.",
            UserWarning,
            stacklevel=2,
        )

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[np.ndarray]]:
        """Return approximate token embeddings for the requested documents.

        Embeddings are reconstructed from stored PQ codes via
        ``approx = coarse_centroid + norm * PQ_residual`` and are therefore
        approximate (PQ lossy compression). When the index was built with
        ``center_dataset=True`` (the default), the dataset mean is added back
        so that the returned embeddings are in the original embedding space.

        Parameters
        ----------
        documents_ids
            Nested list of document string IDs. Each inner list is one group.

        Returns
        -------
        list[list[np.ndarray]]
            Nested list matching the input structure. Each array has shape
            ``(n_tokens, dim)`` in float32.
        """
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before retrieving embeddings."
            )
        self._ensure_loaded()
        self._ensure_mappings()
        return [
            [
                self._index.get_document_embeddings(self._doc_id_to_int[doc_id])
                for doc_id in group
            ]
            for group in documents_ids
        ]

    def __repr__(self) -> str:
        if not self.is_indexed:
            return f"TachiomIndex(path={self.index_path!r}, empty)"
        ef_search_str = (
            f"{round(1.5 * self.k_centroids)} (auto)"
            if self.ef_search is None
            else str(self.ef_search)
        )
        return (
            f"TachiomIndex(\n"
            f"  path={self.index_path!r},\n"
            f"  docs={self._index.len}, tokens={self._index.n_tokens}, dim={self._index.dim},\n"
            f"  — build —\n"
            f"  center_dataset={self.center_dataset}, "
            f"total_centroids={self.total_centroids}, "
            f"tac_n_iter={self.tac_n_iter}, "
            f"tac_micro_threshold={self.tac_micro_threshold}, "
            f"tac_small_threshold={self.tac_small_threshold},\n"
            f"  pq_sample_size={self.pq_sample_size}, "
            f"pq_n_iter={self.pq_n_iter}, "
            f"normalize={self.normalize}, "
            f"pq_seed={self.pq_seed},\n"
            f"  hnsw_m={self.hnsw_m}, "
            f"ef_construction={self.ef_construction},\n"
            f"  — search —\n"
            f"  k_centroids={self.k_centroids}, "
            f"k_docs_to_score={self.k_docs_to_score}, "
            f"ef_search={ef_search_str},\n"
            f"  alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"lambda_={self.lambda_}, "
            f"num_threads={self.num_threads}\n"
            f")"
        )
