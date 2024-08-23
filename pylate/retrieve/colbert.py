import numpy as np
import torch

from ..indexes import Voyager
from ..rank import rerank
from ..utils import iter_batch


class ColBERT:
    """ColBERT retriever.

    Parameters
    ----------
    index:
        The index to use for retrieval.

    Examples
    --------
    >>> from pylate import indexes, models, retrieve

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     device="cpu",
    ... )

    >>> documents_ids = ["1", "2"]

    >>> documents = [
    ...     "fruits are healthy.",
    ...     "fruits are good for health.",
    ... ]

    >>> documents_embeddings = model.encode(
    ...     sentences=documents,
    ...     batch_size=1,
    ...     is_query=False,
    ... )

    >>> index = indexes.Voyager(
    ...     index_folder="test_indexes",
    ...     index_name="colbert",
    ...     override=True,
    ...     embedding_size=128,
    ... )

    >>> index = index.add_documents(
    ...     documents_ids=documents_ids,
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> retriever = retrieve.ColBERT(index=index)

    >>> queries_embeddings = model.encode(
    ...     ["fruits are healthy.", "fruits are good for health."],
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 2

    >>> queries_embeddings = model.encode(
    ...     "fruits are healthy.",
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 1

    """

    def __init__(self, index: Voyager) -> None:
        self.index = index

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_index: int | None = None,
        device: str | None = None,
        batch_size: int = 50,
    ) -> list[list[dict]]:
        """Retrieve documents for a list of queries.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_index
            The number of documents to retrieve from the index. Defaults to `k`.
        device
            The device to use for the embeddings. Defaults to queries_embeddings device.

        """
        reranking_results = []
        for queries_embeddings_batch in iter_batch(
            queries_embeddings,
            batch_size=batch_size,
            desc=f"Retrieving documents (bs={batch_size})",
        ):
            retrieved_elements = self.index(
                queries_embeddings=queries_embeddings_batch,
                k=k if k_index is None else k_index,
            )

            documents_ids = [
                list(
                    set(
                        [
                            document_id
                            for query_token_document_ids in query_documents_ids
                            for document_id in query_token_document_ids
                        ]
                    )
                )
                for query_documents_ids in retrieved_elements["documents_ids"]
            ]

            documents_embeddings = self.index.get_documents_embeddings(documents_ids)

            reranking_results.extend(
                rerank(
                    documents_ids=documents_ids,
                    queries_embeddings=queries_embeddings_batch,
                    documents_embeddings=documents_embeddings,
                    device=device,
                )
            )
        return [query_results[:k] for query_results in reranking_results]