from giga_cherche.indexes.BaseIndex import BaseIndex
from giga_cherche.reranker.ColBERTReranker import ColBERTReranker
from giga_cherche.util import colbert_score
from typing import Optional, List, Union
import torch
from torch import Tensor
import numpy as np


# TODO: define Retriever metaclass
class ColBERTRetriever:
    def __init__(self, index: BaseIndex) -> None:
        self.index = index
        self.reranker = ColBERTReranker(index)

    def retrieve(
        self, queries: List[Union[list, np.ndarray, Tensor]], k: int
    ) -> List[List[str]]:
        # if(isinstance(queries, Tensor)):
        #     queries = queries.cpu().tolist()
        retrieved_elements = self.index.query(queries.cpu().tolist(), int(k / 2))
        batch_doc_ids = [
            list(
                set(
                    [
                        doc_id
                        for token_doc_ids in query_doc_ids
                        for doc_id in token_doc_ids
                    ]
                )
            )[:k]
            for query_doc_ids in retrieved_elements["doc_ids"]
        ]
        return self.reranker.rerank(queries, batch_doc_ids)
