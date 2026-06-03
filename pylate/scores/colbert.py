from __future__ import annotations

import os

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor

# FlashUnsupported / LIKUnsupported are the only exceptions we silently fall
# back on; real bugs (assertions, OOMs, etc.) inside the kernel paths are
# *not* caught.
from ._flash_backend import FlashUnsupported
from ._lik_backend import LIKUnsupported


def _resolve_backend(backend: str | None) -> str:
    """Resolve the effective backend from an explicit arg or the env override.

    The env var `PYLATE_SCORES_BACKEND` is read on every call (not just at
    import time) so users can switch backends at runtime by
    `os.environ["PYLATE_SCORES_BACKEND"] = "torch"` and the next score call
    will pick it up — per @raphaelsty's PR #212 review.
    """
    if backend is None:
        backend = os.environ.get("PYLATE_SCORES_BACKEND", "auto")
    backend = backend.lower()
    if backend not in ("auto", "torch", "flash", "lik"):
        raise ValueError(
            f"backend must be one of 'auto', 'torch', 'flash', 'lik'; got {backend!r}"
        )
    return backend


def _try_flash(backend: str, *tensors: torch.Tensor) -> bool:
    """Decide whether to dispatch to the flash-maxsim backend."""
    if backend not in ("auto", "flash"):
        return False
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
        if backend == "flash":
            raise RuntimeError(
                "backend='flash' requires CUDA tensors; got non-CUDA input"
            )
        return False
    try:
        from . import _flash_backend
    except ImportError:
        if backend == "flash":
            raise
        return False
    if not _flash_backend.is_available():
        if backend == "flash":
            raise RuntimeError(
                "backend='flash' requested but `flash-maxsim` is not installed"
            )
        return False
    return True


def _try_lik(backend: str, *tensors: torch.Tensor) -> bool:
    """Decide whether to dispatch to the late-interaction-kernels backend."""
    if backend not in ("auto", "lik"):
        return False
    on_accelerator = all(
        isinstance(t, torch.Tensor) and (t.is_cuda or t.device.type == "mps")
        for t in tensors
    )
    if not on_accelerator:
        if backend == "lik":
            raise RuntimeError(
                "backend='lik' requires CUDA or MPS tensors; got a CPU/non-tensor input"
            )
        return False
    from . import _lik_backend

    if not _lik_backend.is_available():
        if backend == "lik":
            raise RuntimeError(
                "backend='lik' requested but `late-interaction-kernels` is not "
                'installed; run `pip install "pylate[lik]"`'
            )
        return False
    return True


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    backend: str | None = None,
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
    backend
        Scoring backend: ``"auto"`` (default — tries flash, then LIK, whichever
        is installed and on a supported device, else torch), ``"torch"`` (original
        pure-torch path), ``"flash"`` (requires ``pip install pylate[flash-maxsim]``
        and CUDA inputs; raises otherwise), or ``"lik"`` (requires
        ``pip install "pylate[lik]"`` and CUDA/MPS inputs; raises otherwise).
        Override via env var ``PYLATE_SCORES_BACKEND``.

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
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)

    resolved = _resolve_backend(backend)
    if _try_flash(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._flash_backend import colbert_scores_flash

            return colbert_scores_flash(
                queries_embeddings,
                documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
            )
        except FlashUnsupported:
            if resolved == "flash":
                raise
            # auto: try LIK next, else the torch path below.
    if _try_lik(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._lik_backend import colbert_scores_lik

            return colbert_scores_lik(
                queries_embeddings,
                documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
            )
        except LIKUnsupported:
            if resolved == "lik":
                raise
            # auto: silently fall back to the torch path below.

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


def colbert_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    backend: str | None = None,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    backend
        Scoring backend. See :func:`colbert_scores`.

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
    resolved = _resolve_backend(backend)
    if _try_flash(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._flash_backend import colbert_scores_pairwise_flash

            return colbert_scores_pairwise_flash(
                queries_embeddings, documents_embeddings
            )
        except FlashUnsupported:
            if resolved == "flash":
                raise
    if _try_lik(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._lik_backend import colbert_scores_pairwise_lik

            return colbert_scores_pairwise_lik(queries_embeddings, documents_embeddings)
        except LIKUnsupported:
            if resolved == "lik":
                raise

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
    backend: str | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num candidate documents, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num candidate documents, num tokens documents)
    backend
        Scoring backend. See :func:`colbert_scores`.

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
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)

    resolved = _resolve_backend(backend)
    if _try_flash(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._flash_backend import colbert_kd_scores_flash

            return colbert_kd_scores_flash(
                queries_embeddings,
                documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
            )
        except FlashUnsupported:
            if resolved == "flash":
                raise
    if _try_lik(resolved, queries_embeddings, documents_embeddings):
        try:
            from ._lik_backend import colbert_kd_scores_lik

            return colbert_kd_scores_lik(
                queries_embeddings,
                documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
            )
        except LIKUnsupported:
            if resolved == "lik":
                raise

    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        scores = scores * documents_mask.unsqueeze(2)

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
        backend: str | None = None,
    ) -> torch.Tensor:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)

        D, N, _, _ = documents_embeddings.shape
        # Per-group scores: list of N tensors each of shape (Q_query, D).
        per_group = [
            colbert_scores(
                queries_embeddings,
                documents_embeddings[:, j],
                queries_mask,
                documents_mask[:, j] if documents_mask is not None else None,
                backend=backend,
            )
            for j in range(N)
        ]
        # Stack to (Q_query, D, N) then flatten to (Q_query, D*N) with
        # query-major ordering (doc d's k-th slot at column d*N + k).
        return torch.stack(per_group, dim=2).reshape(-1, D * N)
