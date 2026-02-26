from __future__ import annotations

import numpy as np
import torch
from typing_extensions import TypedDict

from ..scores import colbert_scores
from ..utils import convert_to_tensor as func_convert_to_tensor


class RerankResult(TypedDict):
    """
    Rerank result for ranking.

    Parameters
    ----------
    id
        The document id.
    score
        The document score.
    """

    id: int | str
    score: float


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reshape the embeddings to the correct shape."""
    if isinstance(embeddings, torch.Tensor):
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(dim=0)

    elif isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    return embeddings


def rerank(
    documents_ids: list[list[int | str]],
    queries_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
    documents_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
    device: str = None,
) -> list[list[RerankResult]]:
    """Rerank the documents based on the queries embeddings.

    Parameters
    ----------
    documents_ids
        The documents ids.
    queries_embeddings
        The queries embeddings which is a dictionary of queries and their embeddings.
    documents_embeddings
        The documents embeddings which is a dictionary of documents ids and their embeddings.
    device
        The device to use for the reranking. If None, the device of the queries embeddings will be used.

    Examples
    --------
    >>> from pylate import models, rank

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> queries = [
    ...     "query A",
    ...     "query B",
    ... ]

    >>> documents = [
    ...     ["document A", "document B"],
    ...     ["document 1", "document C", "document B"],
    ... ]

    >>> documents_ids = [
    ...    [1, 2],
    ...    [1, 3, 2],
    ... ]

    >>> queries_embeddings = model.encode(
    ...     queries,
    ...     is_query=True,
    ...     batch_size=1,
    ... )

    >>> documents_embeddings = model.encode(
    ...     documents,
    ...     is_query=False,
    ...     batch_size=1,
    ... )

    >>> reranked_documents = rank.rerank(
    ...     documents_ids=documents_ids,
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> assert isinstance(reranked_documents, list)
    >>> assert len(reranked_documents) == 2
    >>> assert len(reranked_documents[0]) == 2
    >>> assert len(reranked_documents[1]) == 3
    >>> assert isinstance(reranked_documents[0], list)
    >>> assert isinstance(reranked_documents[0][0], dict)
    >>> assert "id" in reranked_documents[0][0]
    >>> assert "score" in reranked_documents[0][0]

    """
    results = []

    queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
    documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)

    for query_embeddings, query_documents_ids, query_documents_embeddings in zip(
        queries_embeddings, documents_ids, documents_embeddings
    ):
        query_embeddings = func_convert_to_tensor(query_embeddings)

        query_documents_embeddings = [
            func_convert_to_tensor(query_document_embeddings)
            for query_document_embeddings in query_documents_embeddings
        ]

        # Pad the documents embeddings
        query_documents_embeddings = torch.nn.utils.rnn.pad_sequence(
            query_documents_embeddings, batch_first=True, padding_value=0
        )

        if device is not None:
            query_embeddings = query_embeddings.to(device)
            query_documents_embeddings = query_documents_embeddings.to(device)
        else:
            query_documents_embeddings = query_documents_embeddings.to(
                query_embeddings.device
            )

        query_scores = colbert_scores(
            queries_embeddings=query_embeddings.unsqueeze(0),
            documents_embeddings=query_documents_embeddings,
        )[0]

        scores, sorted_indices = torch.sort(input=query_scores, descending=True)
        scores = scores.cpu().tolist()

        query_documents = [query_documents_ids[idx] for idx in sorted_indices.tolist()]

        results.append(
            [
                RerankResult(id=doc_id, score=score)
                for doc_id, score in zip(query_documents, scores)
            ]
        )

    return results


def _compute_imputation_scores(
    query_scores: list[list[float]],
    imputation: str,
    percentile: float,
    power_law_multiplier: float,
    device: str,
    is_rectangular: bool | None = None,
) -> torch.Tensor:
    """Compute imputation scores for each query token.

    Parameters
    ----------
    query_scores
        List of length q_tok, where each element is a list of scores.
    imputation
        Imputation strategy: "min", "zero", "mean", "percentile", or "power_law".
    percentile
        Percentile value (0-100) for percentile imputation.
    power_law_multiplier
        Multiplier for k' when extrapolating power-law (e.g., 100 means extrapolate to rank 100*k').
    device
        Device for tensor computation.
    is_rectangular
        If provided, skip the per-row length check and use the given value
        to choose the rectangular fast-path (True) or ragged fallback (False).

    Returns
    -------
    torch.Tensor
        Imputation score for each query token, shape (q_tok,).
    """
    q_tok = len(query_scores)
    allowed_imputations = {"min", "zero", "mean", "percentile", "power_law"}
    if imputation not in allowed_imputations:
        raise ValueError(
            f"Unknown imputation strategy: {imputation}. "
            f"Expected one of: 'min', 'zero', 'mean', 'percentile', 'power_law'."
        )

    if imputation == "zero":
        return torch.zeros(q_tok, dtype=torch.float32, device=device)

    def _to_tensor(values: np.ndarray | list[float]) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def _power_law_row(scores: np.ndarray | list[float]) -> float:
        """Compute per-token power-law imputation score with robust fallbacks."""
        if len(scores) == 0:
            return 0.0

        scores_arr = np.asarray(scores, dtype=np.float64)
        min_score = float(scores_arr.min())
        if len(scores_arr) < 2:
            return min_score

        # Rank 1 corresponds to the highest score.
        sorted_scores = np.sort(scores_arr)[::-1]
        valid_scores = sorted_scores[sorted_scores > 0]
        if len(valid_scores) < 2:
            return min_score

        ranks = np.arange(1, len(valid_scores) + 1, dtype=np.float64)
        try:
            neg_b, log_a = np.polyfit(np.log(ranks), np.log(valid_scores), 1)
            extrapolate_rank = power_law_multiplier * len(sorted_scores)
            log_imputed = log_a + neg_b * np.log(extrapolate_rank)
            imputed = float(np.exp(log_imputed))
            # Keep imputed value in a stable range.
            return max(0.0, min(imputed, min_score))
        except (np.linalg.LinAlgError, ValueError):
            return min_score

    # Rectangular fast path (common for fixed-k retrieval outputs).
    if is_rectangular is None:
        is_rectangular = q_tok > 0 and all(
            len(scores) == len(query_scores[0]) for scores in query_scores
        )
    if is_rectangular:
        # Avoid list-of-ndarray -> tensor conversion warnings by materializing
        # a contiguous ndarray once.
        scores_np = np.asarray(query_scores, dtype=np.float32)
        if scores_np.shape[1] == 0:
            return torch.zeros(q_tok, dtype=torch.float32, device=device)

        if imputation == "min":
            return _to_tensor(scores_np.min(axis=1))
        if imputation == "mean":
            return _to_tensor(scores_np.mean(axis=1))
        if imputation == "percentile":
            return _to_tensor(np.percentile(scores_np, percentile, axis=1))
        return _to_tensor([_power_law_row(row) for row in scores_np])

    # Ragged fallback.
    if imputation == "min":
        return _to_tensor(
            [min(scores) if len(scores) > 0 else 0.0 for scores in query_scores]
        )
    if imputation == "mean":
        return _to_tensor(
            [
                sum(scores) / len(scores) if len(scores) > 0 else 0.0
                for scores in query_scores
            ]
        )
    if imputation == "percentile":
        return _to_tensor(
            [
                float(np.percentile(scores, percentile)) if len(scores) > 0 else 0.0
                for scores in query_scores
            ]
        )
    return _to_tensor([_power_law_row(scores) for scores in query_scores])


def score_xtr(
    query_doc_ids: list[list[str | int]],
    query_scores: list[list[float]],
    k: int,
    device: str = "cpu",
    imputation: str = "min",
    percentile: float = 10.0,
    power_law_multiplier: float = 100.0,
) -> list[RerankResult]:
    """Score documents using XTR (eXact Token Retrieval) scoring.

    XTR scoring differs from ColBERT in that it doesn't do full reranking.
    Instead, it only scores documents using initially retrieved tokens, and
    imputes missing token scores based on the chosen imputation strategy.

    Parameters
    ----------
    query_doc_ids
        List of length q_tok, where each element is a list of k_token document IDs
        retrieved for that query token. Document IDs can be strings or integers.
    query_scores
        List of length q_tok, where each element is a list of k_token scores
        corresponding to the retrieved document IDs.
    k
        Number of top documents to return.
    device
        Device to use for computation ('cpu', 'cuda', etc.).
    imputation
        Strategy for imputing missing scores. Options:
        - "min": Use minimum retrieved score per query token (default, original XTR).
        - "zero": Impute with zero (missing tokens contribute nothing).
        - "mean": Use mean of retrieved scores per query token.
        - "percentile": Use specified percentile of retrieved scores.
        - "power_law": Fit power-law curve to retrieved scores and extrapolate
          to rank (power_law_multiplier * k') as per Lee et al., 2023.
    percentile
        Percentile value (0-100) for percentile imputation. Default is 10.0.
    power_law_multiplier
        Multiplier for k' when extrapolating power-law. Default is 100.0 (extrapolate
        to rank 100*k' as in the original XTR paper).

    Returns
    -------
    list[RerankResult]
        Top-k documents sorted by score (descending).

    Notes
    -----
    The XTR scoring algorithm:
    1. For each document, sum scores across all query tokens
    2. If a document's token wasn't retrieved for a query token, use the
       imputed score based on the chosen strategy
    3. If multiple tokens from the same document were retrieved for a query token,
       use the maximum score

    Examples
    --------
    >>> from pylate.rank import score_xtr
    >>> query_doc_ids = [
    ...     ["doc1", "doc2", "doc3"],  # Retrieved for query token 0
    ...     ["doc2", "doc3", "doc4"],  # Retrieved for query token 1
    ... ]
    >>> query_scores = [
    ...     [0.9, 0.7, 0.5],  # Scores for query token 0
    ...     [0.8, 0.6, 0.4],  # Scores for query token 1
    ... ]
    >>> results = score_xtr(query_doc_ids, query_scores, k=3)
    >>> assert len(results) == 3
    >>> assert results[0]["id"] == "doc2"  # Has high scores for both tokens

    >>> # Using zero imputation
    >>> results_zero = score_xtr(query_doc_ids, query_scores, k=3, imputation="zero")

    >>> # Using power-law imputation
    >>> results_pl = score_xtr(query_doc_ids, query_scores, k=3, imputation="power_law")

    """
    if len(query_doc_ids) != len(query_scores):
        raise ValueError(
            "query_doc_ids and query_scores must have the same number of query tokens. "
            f"Got {len(query_doc_ids)} and {len(query_scores)}."
        )

    q_tok = len(query_doc_ids)

    if q_tok == 0:
        return []

    # Validate lengths and compute total size for pre-allocation.
    sizes = []
    for q_idx, (token_docs, token_scores) in enumerate(
        zip(query_doc_ids, query_scores)
    ):
        if len(token_docs) != len(token_scores):
            raise ValueError(
                "Each query token must have matching document IDs and scores lengths. "
                f"Token {q_idx} has {len(token_docs)} IDs and {len(token_scores)} scores."
            )
        sizes.append(len(token_docs))

    total = sum(sizes)
    if total == 0:
        return []

    # Pre-allocate flat numpy arrays and fill via slice assignment (avoids
    # repeated list resizing and gives faster torch.from_numpy conversion).
    all_scores_np = np.empty(total, dtype=np.float32)
    q_tok_indices_np = np.empty(total, dtype=np.int64)

    # Determine doc-ID type from the first non-empty token list.
    doc_id_is_string = isinstance(
        query_doc_ids[0][0]
        if sizes[0] > 0
        else query_doc_ids[next(i for i, s in enumerate(sizes) if s > 0)][0],
        str,
    )

    if doc_id_is_string:
        # Build string→int mapping while filling the flat arrays in one pass.
        # Preserves insertion order for deterministic tie handling.
        doc_id_to_int: dict[str, int] = {}
        all_doc_ids_np = np.empty(total, dtype=np.int64)
        offset = 0
        for q_idx, (token_docs, token_scores) in enumerate(
            zip(query_doc_ids, query_scores)
        ):
            n = sizes[q_idx]
            if n == 0:
                continue
            for i, did in enumerate(token_docs):
                if did not in doc_id_to_int:
                    doc_id_to_int[did] = len(doc_id_to_int)
                all_doc_ids_np[offset + i] = doc_id_to_int[did]
            all_scores_np[offset : offset + n] = token_scores
            q_tok_indices_np[offset : offset + n] = q_idx
            offset += n

        # Invert the mapping for final output.
        unique_doc_id_strings = list(doc_id_to_int.keys())
        num_docs = len(unique_doc_id_strings)

        # The integer IDs are already in [0, num_docs), so inverse_indices
        # IS the doc-id tensor — no torch.unique() needed.
        inverse_indices = torch.from_numpy(all_doc_ids_np).to(device=device)
    else:
        # Integer doc IDs — need torch.unique to discover the unique set.
        all_doc_ids_list: list[int] = []
        offset = 0
        for q_idx, (token_docs, token_scores) in enumerate(
            zip(query_doc_ids, query_scores)
        ):
            n = sizes[q_idx]
            if n == 0:
                continue
            all_doc_ids_list.extend(token_docs)
            all_scores_np[offset : offset + n] = token_scores
            q_tok_indices_np[offset : offset + n] = q_idx
            offset += n

        all_doc_ids_t = torch.tensor(all_doc_ids_list, dtype=torch.long, device=device)
        unique_doc_ids, inverse_indices = torch.unique(
            all_doc_ids_t, return_inverse=True, sorted=False
        )
        num_docs = len(unique_doc_ids)

    all_scores_t = torch.from_numpy(all_scores_np).to(device=device)
    q_tok_indices_t = torch.from_numpy(q_tok_indices_np).to(device=device)

    # Compute imputation scores based on chosen strategy.
    # We already know whether all token lists are the same length from `sizes`.
    rect_hint = len(set(sizes)) <= 1
    imputation_scores = _compute_imputation_scores(
        query_scores=query_scores,
        imputation=imputation,
        percentile=percentile,
        power_law_multiplier=power_law_multiplier,
        device=device,
        is_rectangular=rect_hint,
    )  # Shape: (q_tok,)

    # Step 1: Compute max actual score per (doc, query_token) pair
    # Initialize with -inf so we can detect which pairs have no retrieved score
    NEG_INF = float("-inf")
    doc_scores = torch.full(
        (num_docs, q_tok), NEG_INF, dtype=torch.float32, device=device
    )

    # Flatten for 1D scatter, then reshape
    doc_scores_flat = doc_scores.reshape(-1)
    flat_indices = inverse_indices * q_tok + q_tok_indices_t

    # Use scatter_reduce with reduce='amax' to keep max score when multiple tokens
    # from the same document are retrieved for a single query token
    doc_scores_flat.scatter_reduce_(
        0, flat_indices, all_scores_t, reduce="amax", include_self=False
    )
    doc_scores = doc_scores_flat.reshape(num_docs, q_tok)

    # Step 2: Replace -inf (no retrieved score) with imputation scores
    missing_mask = doc_scores == NEG_INF
    doc_scores = torch.where(missing_mask, imputation_scores, doc_scores)

    # Sum across query tokens to get final document scores
    final_scores = doc_scores.sum(dim=1)

    # Get top k documents
    top_k_scores, top_k_indices = torch.topk(
        final_scores, k=min(k, num_docs), largest=True
    )

    # Bulk-convert to Python lists (single C-to-Python transition).
    top_k_scores_list = top_k_scores.tolist()

    if doc_id_is_string:
        # top_k_indices index into [0, num_docs) which maps directly to
        # unique_doc_id_strings -- no intermediate tensor lookup needed.
        top_k_idx_list = top_k_indices.tolist()
        return [
            RerankResult(id=unique_doc_id_strings[idx], score=score)
            for idx, score in zip(top_k_idx_list, top_k_scores_list)
        ]

    top_k_doc_ids_list = unique_doc_ids[top_k_indices].tolist()
    return [
        RerankResult(id=doc_id, score=score)
        for doc_id, score in zip(top_k_doc_ids_list, top_k_scores_list)
    ]
