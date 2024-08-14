import numpy as np
import torch

from ..indexes import Base as BaseIndex
from ..rank import rerank


class ColBERT:
    """ColBERT retriever.


    Examples
    --------

    """

    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    def retrieve(
        self, queries: list[list | np.ndarray | torch.Tensor], k: int
    ) -> list[list[str]]:
        # if(isinstance(queries, Tensor)):
        #     queries = queries.cpu().tolist()
        retrieved_elements = self.index.query(queries_embeddings=queries, k=k // 2)

        batch_documents_ids = [
            list(
                set(
                    [
                        doc_id
                        for token_doc_ids in query_doc_ids
                        for doc_id in token_doc_ids
                    ]
                )
            )
            for query_doc_ids in retrieved_elements["doc_ids"]
        ]
        batch_documents_embeddings = self.index.get_documents_embeddings(
            batch_documents_ids
        )
        reranking_results = rerank(
            documents_ids=batch_documents_ids,
            queries_embeddings=queries,
            documents_embeddings=batch_documents_embeddings,
        )

        # Only keep the top-k elements for each query
        reranking_results = [query_results[:k] for query_results in reranking_results]

        return reranking_results
