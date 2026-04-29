from __future__ import annotations

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


class XTRScores:
    """XTR contrastive scoring with global top-k token retrieval.

    For each query token, the top-k matches are selected globally across all
    ``Q*N`` in-batch document tokens (simulating retrieval from an index).
    Returns the full ``(Q, Q*N)`` cross-product score matrix with query-major
    ordering: ``scores[i, j*N + k]`` is query ``i`` against query ``j``'s
    ``k``-th document. The positive for query ``i`` sits at column ``i*N``.

    Parameters
    ----------
    k
        Number of top token matches to retain per query token across all Q*N documents.
    document_chunk_size
        If set, the matmul + ``masked_fill`` phase is iterated over
        ``document_chunk_size`` docs at a time (out of ``Q*N`` total). The
        resulting chunks are concatenated before the global top-k, so scoring
        semantics are unchanged. Useful to trim the transient matmul peak at
        large effective batch sizes. Default ``None`` runs the full matmul
        in one shot.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1., 0.], [0., 0.]],
    ...     [[0., 1.], [0., 0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
    ...     [[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]],
    ... ])

    >>> scores = XTRScores(k=2)(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )
    >>> scores.shape
    torch.Size([2, 4])

    Notes
    -----
    Adapted from PrimeQA (Copyright 2026 IBM PrimeQA Authors, licensed under
    the Apache License, Version 2.0). Changes from the original implementation:

    - Extricated the scoring function from the end-to-end modeling class that
      also handled contrastive loss computation.
    - Fixed a bug in the original implementation where the alignment mask was
      not being applied correctly.

    References
    ----------
    - [PrimeQA XTR](https://github.com/primeqa/primeqa/blob/bb9385fa129a0dbb3c7aae96ad3c782913f8280d/primeqa/ir/dense/xtr_top/xtr/modeling/XTR.py)

    """

    def __init__(self, k: int = 256, document_chunk_size: int | None = None):
        self.k = k
        self.document_chunk_size = document_chunk_size

    def compile(self, *args, **kwargs):
        # Shadowing the bound method with an instance attribute works here
        # because _score is looked up via normal attribute resolution
        # (unlike __call__, which Python resolves on the type).
        self._score = torch.compile(self._score, *args, **kwargs)

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)
        return self._score(
            queries_embeddings, documents_embeddings, queries_mask, documents_mask
        )

    def _score(
        self, queries_embeddings, documents_embeddings, queries_mask, documents_mask
    ):
        Qb, Qt, H = queries_embeddings.shape
        D, N, Dt, _ = documents_embeddings.shape
        Db = D * N

        docs_flat = documents_embeddings.reshape(Db, Dt, H)
        Q_flat = queries_embeddings.reshape(Qb * Qt, H)
        docs_mask_flat = (
            documents_mask.reshape(Db, Dt) if documents_mask is not None else None
        )

        doc_chunk = self.document_chunk_size
        if doc_chunk is None or doc_chunk >= Db:
            # Single large matmul — tensor core friendly.
            D_flat = docs_flat.reshape(Db * Dt, H).T
            scores = (Q_flat @ D_flat).view(Qb, Qt, Db, Dt)
            if docs_mask_flat is not None:
                scores = scores.masked_fill(
                    ~docs_mask_flat.bool().unsqueeze(0).unsqueeze(0),
                    torch.finfo(scores.dtype).min,
                )
        else:
            # Chunk the doc axis: each chunk's matmul + masked_fill holds only
            # a (Qb, Qt, chunk, Dt) intermediate. Chunks are concatenated
            # before the global top-k so scoring semantics are unchanged.
            score_chunks = []
            for d_start in range(0, Db, doc_chunk):
                d_end = min(d_start + doc_chunk, Db)
                db = d_end - d_start
                chunk_D_flat = docs_flat[d_start:d_end].reshape(db * Dt, H).T
                chunk_scores = (Q_flat @ chunk_D_flat).view(Qb, Qt, db, Dt)
                if docs_mask_flat is not None:
                    chunk_mask = docs_mask_flat[d_start:d_end]
                    chunk_scores = chunk_scores.masked_fill(
                        ~chunk_mask.bool().unsqueeze(0).unsqueeze(0),
                        torch.finfo(chunk_scores.dtype).min,
                    )
                score_chunks.append(chunk_scores)
            scores = torch.cat(score_chunks, dim=2)

        # Global top-k across the full Db*Dt pool.
        clubbed = scores.flatten(2, 3)  # (Qb, Qt, Db*Dt)
        _, indices = clubbed.topk(self.k, dim=-1, sorted=False)
        mask = torch.zeros_like(clubbed, dtype=torch.bool).scatter_(-1, indices, True)
        masked = clubbed * mask
        topk_scores_max = masked.view(Qb, Qt, Db, Dt).max(dim=-1).values  # (Qb, Qt, Db)

        if queries_mask is not None:
            topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)

        scores_sum = topk_scores_max.sum(dim=1)  # (Qb, Db)
        Z = topk_scores_max.gt(0).float().sum(dim=1).clamp_(min=1e-3)
        return (scores_sum / Z).float()


class XTRKDScores(XTRScores):
    """XTR scores for knowledge distillation. Same global top-k scoring as
    :class:`XTRScores`, but returns each query's own N-way document scores
    ``(Q, N)`` instead of the full ``(Q, Q*N)`` cross-product — matching the
    interface expected by :class:`~pylate.losses.Distillation`.
    """

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        documents_embeddings = convert_to_tensor(documents_embeddings)
        Q, N = documents_embeddings.shape[:2]

        # Full cross-product scores: (Q, Q*N)
        all_scores = super().__call__(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )

        # Slice out each query's own N documents
        idx = torch.arange(Q, device=all_scores.device).unsqueeze(1) * N + torch.arange(
            N, device=all_scores.device
        )
        return all_scores.gather(1, idx)


# Default instances — backward compatible as bare callables
xtr_scores = XTRScores()
xtr_kd_scores = XTRKDScores()
