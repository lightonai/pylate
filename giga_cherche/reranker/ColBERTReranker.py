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

    def rerank(self, queries: Union[list, np.ndarray, Tensor], doc_ids: List[List[str]]) -> List[str]:
        documents_embeddings = self.index.get_doc_embeddings(doc_ids)
        documents_embeddings = [torch.tensor(embeddings, dtype=torch.float32, device=queries.device) for embeddings in documents_embeddings]
        documents_embeddings = torch.nn.utils.rnn.pad_sequence(documents_embeddings, batch_first=True, padding_value=0)
        documents_attention_mask = (documents_embeddings.sum(dim=-1) != 0).float()
        scores = colbert_score(queries, documents_embeddings, documents_attention_mask)
        print(scores)