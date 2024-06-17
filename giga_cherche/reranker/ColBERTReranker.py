from giga_cherche.indexes.BaseIndex import BaseIndex
from giga_cherche.util import colbert_score
from typing import Optional, List, Union
import torch
from torch import Tensor
import numpy as np

#TODO: define Reranker metaclass
class ColBERTReranker:
    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    def rerank(self, queries: List[Union[list, np.ndarray, Tensor]], batch_doc_ids: List[List[str]]) -> List[List[str]]:
        batch_documents_embeddings = self.index.get_doc_embeddings(batch_doc_ids)
        # documents_embeddings = [self.index.get_doc_embeddings(query_doc_ids) for query_doc_ids in doc_ids]
        scores = []
        reranked_doc_ids = []
        for query, query_documents_embeddings, query_doc_ids in zip(queries,batch_documents_embeddings, batch_doc_ids):
            documents_embeddings = [torch.tensor(embeddings, dtype=torch.float32, device=queries.device) for embeddings in query_documents_embeddings]
            documents_embeddings = torch.nn.utils.rnn.pad_sequence(documents_embeddings, batch_first=True, padding_value=0)
            documents_attention_mask = (documents_embeddings.sum(dim=-1) != 0).float()
      
            query_scores = colbert_score(query.unsqueeze(0), documents_embeddings, documents_attention_mask)[0]
            sorted_indices = torch.argsort(query_scores, descending=True)
            reranked_query_doc_ids = [query_doc_ids[idx] for idx in sorted_indices.tolist()]
            reranked_doc_ids.append(reranked_query_doc_ids)
            scores.append(query_scores.cpu().tolist())
        # Reorder doc_ids based on the scores
        # reranked_doc_ids = [
        # [query_doc_ids[idx] for idx in torch.argsort(query_scores, descending=True).tolist()]
        # for query_scores, query_doc_ids in zip(scores, batch_doc_ids)
        # ]
        # reranked_doc_ids = []
        # for query_scores, query_doc_ids in zip(scores, batch_doc_ids):
        #     sorted_indices = torch.argsort(query_scores[0], descending=True)
        #     reranked_query_doc_ids = [query_doc_ids[idx] for idx in sorted_indices.tolist()]
        #     reranked_doc_ids.append(reranked_query_doc_ids)

        # doc_ids = [[doc_id for doc_id, _ in sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)] for doc_ids in batch_doc_ids]
        return {"doc_ids": reranked_doc_ids, "scores": scores}

        # doc_ids = [doc_ids[torch.argsort(scores)] for doc_ids, scores in zip(batch_doc_ids, scores)]
        # print(doc_ids)

        # documents_embeddings = [torch.tensor(embeddings, dtype=torch.float32, device=queries.device) for embeddings in documents_embeddings]
        # documents_embeddings = torch.nn.utils.rnn.pad_sequence(documents_embeddings, batch_first=True, padding_value=0)
        # documents_attention_mask = (documents_embeddings.sum(dim=-1) != 0).float()
        # scores = colbert_score(queries, documents_embeddings, documents_attention_mask)
        # print(scores)