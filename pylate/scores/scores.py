import numpy as np
import torch
import torch.nn.functional as F

from ..utils.tensor import convert_to_tensor


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

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

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> scores
    tensor([[  10.,  100., 1000.],
            [  20.,  200., 2000.],
            [  30.,  300., 3000.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)
    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if mask is not None:
        mask = convert_to_tensor(mask)
        scores = scores * mask.unsqueeze(0).unsqueeze(2)

    return scores.max(axis=-1).values.sum(axis=-1)


def colbert_scores_top_p(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    mask: torch.Tensor = None,
    p: float = 0.001,
) -> torch.Tensor:
    """
    Computes the ColBERT scores between queries and documents embeddings using top-p sampling.

    Parameters
    ----------
    queries_embeddings : torch.Tensor
        The queries embeddings. Shape: (batch_size, num_tokens_queries, embedding_size)
    documents_embeddings : torch.Tensor
        The documents embeddings. Shape: (batch_size, num_tokens_documents, embedding_size)
    mask : torch.Tensor, optional
        Optional mask tensor. Default is None.
    p : float, optional
        The cumulative probability threshold for top-p sampling. Default is 0.005.

    Returns
    -------
    torch.Tensor
        The computed scores. Shape: (batch_size, batch_size)
    """
    # Compute similarity scores
    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if mask is not None:
        scores = scores * mask.unsqueeze(0).unsqueeze(2)

    # Apply softmax along the document tokens dimension
    probs = F.softmax(scores, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # print(cumulative_probs[0, 0])

    # Create a mask for probabilities above the threshold
    mask = cumulative_probs <= p

    # Ensure at least one token is selected
    mask[..., 0] = True

    # Apply the mask to the sorted probabilities
    masked_probs = sorted_probs * mask.float()

    # Compute the sum of the selected probabilities
    selected_probs_sum = masked_probs.sum(dim=-1)

    # Sum over query tokens
    final_scores = selected_probs_sum.sum(dim=-1)

    return final_scores


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
    mask: torch.Tensor = None,
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
    >>> mask = torch.tensor([
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 0.]],
    ... ])
    >>> colbert_kd_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     mask=mask
    ... )
    tensor([[  10.,   20.,   30.],
            [ 200.,  400.,  600.],
            [3000., 6000., 30.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )
    if mask is not None:
        mask = convert_to_tensor(mask)
        scores = scores * mask.unsqueeze(2)

    return scores.max(axis=-1).values.sum(axis=-1)


def colbert_kd_scores_top_p(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    mask: torch.Tensor = None,
    p: float = 0.002,
) -> torch.Tensor:
    """
    Computes the ColBERT scores between queries and documents embeddings using top-p sampling.
    This scoring function is dedicated to the knowledge distillation pipeline.

    Parameters
    ----------
    queries_embeddings : torch.Tensor
        The queries embeddings. Shape: (batch_size, num_tokens_queries, embedding_size)
    documents_embeddings : torch.Tensor
        The documents embeddings. Shape: (batch_size, num_documents, num_tokens_documents, embedding_size)
    mask : torch.Tensor, optional
        Optional mask tensor. Shape: (batch_size, num_documents, num_tokens_documents). Default is None.
    p : float, optional
        The cumulative probability threshold for top-p sampling. Default is 0.9.

    Returns
    -------
    torch.Tensor
        The computed scores. Shape: (batch_size, num_documents)
    """
    # Compute similarity scores
    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if mask is not None:
        scores = scores * mask.unsqueeze(2)

    # Apply softmax along the document tokens dimension
    probs = F.softmax(scores, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Broadcast p to match the shape of cumulative_probs
    # p_broadcast = torch.full_like(cumulative_probs, p)
    # Create a mask for probabilities above the threshold
    top_p_mask = cumulative_probs <= p

    # Ensure at least one token is selected
    top_p_mask[..., 0] = True

    # Apply the mask to the sorted probabilities
    masked_probs = sorted_probs * top_p_mask.float()

    # Compute the sum of the selected probabilities
    selected_probs_sum = masked_probs.sum(dim=-1)

    # Sum over query tokens
    final_scores = selected_probs_sum.sum(dim=-1)

    return final_scores
