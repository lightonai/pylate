from __future__ import annotations

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num tokens documents)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [10.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> documents_mask = torch.tensor([
    ...     [1., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 1.],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )

    >>> scores
    tensor([[  10.,  10., 1000.],
            [  20.,  20., 2000.],
            [  0.,  0., 0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


def colbert_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [1.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> scores = colbert_scores_pairwise(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> scores
    tensor([  10.,  200., 3000.])

    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())

    return torch.stack(scores, dim=0)


def colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[10.], [0.], [1.]], [[20.], [0.], [1.]], [[30.], [0.], [1.]]],
    ...     [[[0.], [100.], [1.]], [[0.], [200.], [1.]], [[0.], [300.], [1.]]],
    ...     [[[1.], [0.], [1000.]], [[1.], [0.], [2000.]], [[10.], [0.], [3000.]]],
    ... ])
    >>> documents_mask = torch.tensor([
    ...     [[0., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])
    >>> colbert_kd_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )
    tensor([[ 1.,  20.,  30.],
            [200., 400., 600.],
            [  0.,   0.,   0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        mask = convert_to_tensor(documents_mask)
        scores = scores * mask.unsqueeze(2)

    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


class ColBERTScores:
    """ColBERT contrastive scoring.

    Takes ``(Q_query, Qt, H)`` queries and ``(Q_doc, N, Dt, H)`` stacked
    per-query document groups and returns the full ``(Q_query, Q_doc * N)``
    score matrix with query-major ordering: ``scores[i, j*N + k]`` is the
    score of query ``i`` against the i-th entry of doc group ``j``'s ``k``-th
    slot. When called with matched ``Q_query == Q_doc``, the positive for
    query ``i`` sits at column ``i*N``.

    The document dimension is iterated group-by-group internally so that only
    one ``(Q_query, Q_doc, Qt, Dt)`` intermediate is live at a time.
    """

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)

        D, N, Dt, H = documents_embeddings.shape
        # Per-group scores: list of N tensors each of shape (Q_query, D).
        per_group = [
            colbert_scores(
                queries_embeddings,
                documents_embeddings[:, j],
                queries_mask,
                documents_mask[:, j] if documents_mask is not None else None,
            )
            for j in range(N)
        ]
        # Stack to (Q_query, D, N) then flatten to (Q_query, D*N) with
        # query-major ordering (doc d's k-th slot at column d*N + k).
        return torch.stack(per_group, dim=2).reshape(-1, D * N)


# Adapted from PrimeQA (https://github.com/primeqa/primeqa branch:xtr)
# Specifically: https://github.com/primeqa/primeqa/blob/bb9385fa129a0dbb3c7aae96ad3c782913f8280d/primeqa/ir/dense/xtr_top/xtr/modeling/XTR.py

#   Copyright 2026 IBM PrimeQA Authors
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Changes:
# - extricated scoring function from E2E modeling class that also handled contrastive loss computation.
# - fixed a bug in the original implementation where the alignment mask was not being applied correctly.


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
        Qb = queries_embeddings.shape[0]
        D, N = documents_embeddings.shape[:2]
        Db = D * N
        Qt = queries_embeddings.shape[1]
        Dt = documents_embeddings.shape[-2]
        H = queries_embeddings.shape[-1]

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


class XTRKDScores:
    """XTR scores for knowledge distillation. Same global top-k scoring as
    :class:`XTRScores`, but returns only each query's own N-way document scores
    to match the ``(Q, N)`` interface expected by :class:`~pylate.losses.Distillation`.

    Parameters
    ----------
    k
        Number of top token matches to retain per query token.
    """

    def __init__(self, k: int = 256):
        self._xtr_scores = XTRScores(k=k)

    @property
    def k(self):
        return self._xtr_scores.k

    @k.setter
    def k(self, value):
        self._xtr_scores.k = value

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
        all_scores = self._xtr_scores(
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
