from __future__ import annotations

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


def full_batch_scoring(fn):
    """Mark a scoring function as requiring all documents at once (no chunking).

    Score functions decorated with this will receive documents as a single
    ``(Q, N, Dt, H)`` tensor rather than being called in a per-group loop.
    The ``Contrastive`` and ``CachedContrastive`` losses check for
    ``getattr(score_metric, 'requires_full_batch', False)`` to decide
    which path to take.
    """
    fn.requires_full_batch = True
    return fn


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


@full_batch_scoring
def xtr_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k: int = 128,
) -> torch.Tensor:
    """Computes XTR scores between queries and documents using global top-k token retrieval.

    Each query's top-k token matches are selected globally across all Q*N documents in the
    batch, simulating retrieval from an index. Returns the full (Q, Q*N) cross-product score
    matrix so that all in-batch documents compete as negatives.

    Parameters
    ----------
    queries_embeddings
        Query token embeddings. Shape: (Q, Qt, H)
    documents_embeddings
        Document token embeddings grouped per query. Shape: (Q, N, Dt, H)
    queries_mask
        Attention mask for queries. Shape: (Q, Qt). Pass None when using query expansion so
        that MASK/expansion tokens are not zeroed out.
    documents_mask
        Attention mask for documents. Shape: (Q, N, Dt)
    k
        Number of top token matches to retain per query token across all Q*N documents.

    Returns
    -------
    torch.Tensor
        Scores of shape (Q, Q*N). For contrastive training with N nway docs per query, the
        positive for query i is at column i*N.

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

    >>> scores = xtr_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     k=2,
    ... )
    >>> scores.shape
    torch.Size([2, 4])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    Qb = queries_embeddings.shape[0]
    Dq, N = documents_embeddings.shape[:2]

    # (Dq, N, Dt, H) → (Dq*N, Dt, H)
    docs_flat = documents_embeddings.view(Dq * N, *documents_embeddings.shape[2:])

    # All-pair token scores: (Qb, Dq*N, Qt, Dt)
    scores = queries_embeddings.unsqueeze(1) @ docs_flat.transpose(1, 2).unsqueeze(0)

    if documents_mask is not None:
        # (Dq, N, Dt) → (Dq*N, Dt), expand to (Qb, Dq*N, Dt)
        docs_mask_flat = documents_mask.view(Dq * N, -1)
        D_mask = docs_mask_flat.unsqueeze(0).expand(Qb, -1, -1)
        scores.transpose(2, 3)[~D_mask.bool()] = -99999

    Qb, Db, Qt, Dt = scores.shape

    # (Qb, Qt, Dq*N*Dt) — club all doc tokens together per query token
    clubbed = scores.permute(0, 2, 1, 3).flatten(2, 3)

    _, topk_indices = clubbed.topk(k, dim=-1)  # (Qb, Qt, k)

    # Zero out all non-top-k positions
    alignment_mask = torch.ones_like(clubbed, dtype=torch.bool)
    alignment_mask.scatter_(-1, topk_indices, False)
    masked = clubbed.masked_fill(alignment_mask, 0)

    # (Qb, Qt, Dq*N, Dt) → max over Dt → (Qb, Qt, Dq*N)
    topk_scores_max = masked.view(Qb, Qt, Db, Dt).max(dim=-1).values

    if queries_mask is not None:
        topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)

    # Z: number of non-zero query-token contributions per doc, clamped for stability
    Z = (topk_scores_max > 0.0).float().sum(dim=1).clamp(min=1e-3)  # (Qb, Dq*N)

    return (1.0 / Z) * topk_scores_max.sum(dim=1)  # (Qb, Dq*N)


def xtr_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k: int = 128,
) -> torch.Tensor:
    """XTR scores for knowledge distillation. Same global top-k scoring as
    :func:`xtr_scores`, but returns only each query's own N-way document scores
    to match the ``(Q, N)`` interface expected by :class:`~pylate.losses.Distillation`.

    Parameters
    ----------
    queries_embeddings
        Query token embeddings. Shape: (Q, Qt, H)
    documents_embeddings
        Document token embeddings grouped per query. Shape: (Q, N, Dt, H)
    queries_mask
        Attention mask for queries. Shape: (Q, Qt)
    documents_mask
        Attention mask for documents. Shape: (Q, N, Dt)
    k
        Number of top token matches to retain per query token.

    Returns
    -------
    torch.Tensor
        Scores of shape (Q, N).
    """
    documents_embeddings = convert_to_tensor(documents_embeddings)
    Q, N = documents_embeddings.shape[:2]

    # Full cross-product scores: (Q, Q*N)
    all_scores = xtr_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        k=k,
    )

    # Slice out each query's own N documents
    idx = torch.arange(Q, device=all_scores.device).unsqueeze(1) * N + torch.arange(
        N, device=all_scores.device
    )
    return all_scores.gather(1, idx)
