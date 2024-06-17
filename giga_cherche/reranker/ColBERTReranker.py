from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from giga_cherche.indexes.BaseIndex import BaseIndex
from giga_cherche.scores.colbert_score import colbert_score


# TODO: define Reranker metaclass
class ColBERTReranker:
    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    def rerank(
        self,
        queries: List[Union[list, np.ndarray, Tensor]],
        batch_doc_ids: List[List[str]],
    ) -> List[List[str]]:
        batch_documents_embeddings = self.index.get_doc_embeddings(batch_doc_ids)
        # documents_embeddings = [self.index.get_doc_embeddings(query_doc_ids) for query_doc_ids in doc_ids]
        reranked_doc_ids = []
        reranked_scores = []
        # We do not batch queries to prevent memory overhead (computing the scores could be intensive), prevent unecessary padding of documents to the largest documents in the batch and also because the number of documents per query is not fixed.
        for query, query_documents_embeddings, query_doc_ids in zip(
            queries, batch_documents_embeddings, batch_doc_ids
        ):
            documents_embeddings = [
                torch.tensor(embeddings, dtype=torch.float32, device=queries.device)
                for embeddings in query_documents_embeddings
            ]
            documents_embeddings = torch.nn.utils.rnn.pad_sequence(
                documents_embeddings, batch_first=True, padding_value=0
            )
            query_scores = colbert_score(query.unsqueeze(0), documents_embeddings)[0]
            reranked_query_scores, sorted_indices = torch.sort(
                query_scores, descending=True
            )

            # Reorder doc_ids based on the scores
            reranked_query_doc_ids = [
                query_doc_ids[idx] for idx in sorted_indices.tolist()
            ]
            reranked_doc_ids.append(reranked_query_doc_ids)
            reranked_scores.append(reranked_query_scores.cpu().tolist())

        return {"doc_ids": reranked_doc_ids, "scores": reranked_scores}
