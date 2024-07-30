import numpy as np
import torch

from ..indexes import Base as BaseIndex
from ..rerank import ColBERT as ColBERTReranker


class ColBERT:
    """ColBERT retriever.


    Examples
    --------



    """

    def __init__(self, index: BaseIndex) -> None:
        self.index = index
        self.reranker = ColBERTReranker(index=index)

    def retrieve(
        self, queries: list[list | np.ndarray | torch.Tensor], k: int
    ) -> list[list[str]]:
        # if(isinstance(queries, Tensor)):
        #     queries = queries.cpu().tolist()
        retrieved_elements = self.index.query(queries_embeddings=queries, k=k // 2)

        batch_doc_ids = [
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

        reranking_results = self.reranker.rerank(
            queries=queries, batch_doc_ids=batch_doc_ids
        )

        # Only keep the top-k elements for each query
        reranking_results = [query_results[:k] for query_results in reranking_results]

        return reranking_results
