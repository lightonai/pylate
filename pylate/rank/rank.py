import numpy as np
import torch

from ..scores import colbert_scores
from ..utils import convert_to_tensor as func_convert_to_tensor


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
    retrieved_scores,
    device: str = None,
) -> list[list[dict[str, float]]]:
    """Rerank the documents based on the queries embeddings.

    Parameters
    ----------
    queries
        The queries.
    documents_ids
        The documents ids.
    queries_embeddings
        The queries embeddings which is a dictionary of queries and their embeddings.
    documents_embeddings
        The documents embeddings which is a dictionary of documents ids and their embeddings.

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
    retrieved_scores = torch.Tensor(retrieved_scores)

    for (
        query_embeddings,
        query_documents_ids,
        query_documents_embeddings,
        query_retrieved_scores,
    ) in zip(queries_embeddings, documents_ids, documents_embeddings, retrieved_scores):
        query_embeddings = func_convert_to_tensor(query_embeddings)

        query_documents_embeddings = [
            func_convert_to_tensor(query_document_embeddings)
            for query_document_embeddings in query_documents_embeddings
        ]

        # Pad the documents embeddings
        query_documents_embeddings = torch.nn.utils.rnn.pad_sequence(
            query_documents_embeddings, batch_first=True, padding_value=0
        )
        # print("quyerydoc")
        # print(query_documents_embeddings.shape)

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
            retrieved_scores=query_retrieved_scores,
        )[0]

        scores, sorted_indices = torch.sort(input=query_scores, descending=True)
        scores = scores.cpu().tolist()

        query_documents = [query_documents_ids[idx] for idx in sorted_indices.tolist()]

        results.append(
            [
                {"id": doc_id, "score": score}
                for doc_id, score in zip(query_documents, scores)
            ]
        )

    return results
