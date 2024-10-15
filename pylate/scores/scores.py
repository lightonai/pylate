import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    retrieved_scores: list | np.ndarray | torch.Tensor,
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
    max_scores = scores.max(axis=-1).values  # Shape [1, num_candidate_docs, 32]
    # print(max_scores[0][0])

    # Add the max_scores to the retrieved_scores to be considered for max/min if they were not orignally retrieved
    retrieved_scores = torch.cat(
        (retrieved_scores, max_scores.squeeze(0).transpose(0, 1)), dim=1
    )
    # Step 1: Extract min and max scores from the second tensor
    min_retrieved_scores, _ = retrieved_scores.min(dim=1)  # Shape: [32]
    max_retrieved_scores, _ = retrieved_scores.max(dim=1)  # Shape: [32]
    max_retrieved_scores = max_retrieved_scores.view(1, 1, -1)
    min_retrieved_scores = min_retrieved_scores.view(1, 1, -1)
    # print(max_retrieved_scores.shape)
    # print(max_scores.shape)
    # Step 2 & 3: Normalize the first tensor
    normalized_scores = (max_scores - min_retrieved_scores) / (
        max_retrieved_scores - min_retrieved_scores
    )  # * max_scores
    # normalized_scores = torch.nn.functional.softmax(normalized_scores, dim=-1)
    # normalized_scores = torch.relu(normalized_scores)

    # # Step 1: Extract mean scores from retrieved_scores
    # mean_retrieved_scores = retrieved_scores.mean(dim=1).view(
    #     1, 1, -1
    # )  # Shape: [1, 1, 32]

    # # std_retrieved_scores = retrieved_scores.std(dim=1).view(
    # #     1, 1, -1
    # # )  # Shape: [1, 1, 32]

    # # Step 2 & 3: Normalize the scores by subtracting the mean
    # normalized_scores = max_scores - mean_retrieved_scores
    # # normalized_scores = (max_scores - mean_retrieved_scores) / (
    # #     std_retrieved_scores + 1e-8
    # # )
    # # normalized_scores = torch.relu(max_scores - mean_retrieved_scores)

    # # print(max_scores - normalized_scores)
    # print(normalized_scores[0][0])
    # print(normalized_scores.sum(axis=-1))
    return normalized_scores.sum(axis=-1)
    # return scores.max(axis=-1).values.sum(axis=-1)


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
