import numpy as np
import torch

from ..indexes import Base as BaseIndex
from ..scores import colbert_score

__all__ = ["ColBERT"]


class ColBERT:
    """Rerank

    Parameters

    """

    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    def rerank(
        self,
        queries: list[list | np.ndarray | torch.Tensor],
        batch_doc_ids: list[list[str]],
    ) -> list[list[str]]:
        batch_documents_embeddings = self.index.get_docs_embeddings(batch_doc_ids)
        # documents_embeddings = [self.index.get_doc_embeddings(query_doc_ids) for query_doc_ids in doc_ids]
        reranked_doc_ids = []
        reranked_scores = []
        res = []
        # If fed a list of numpy arrays, convert them to torch.Tensors
        if not isinstance(queries[0], torch.Tensor):
            queries = torch.from_numpy(np.array(queries, dtype=np.float32))
        # We do not batch queries to prevent memory overhead (computing the scores could be intensive), prevent unecessary padding of documents to the largest documents in the batch and also because the number of documents per query is not fixed.
        for query, query_documents_embeddings, query_doc_ids in zip(
            queries, batch_documents_embeddings, batch_doc_ids
        ):
            documents_embeddings = [
                torch.tensor(embeddings, dtype=torch.float32, device=query.device)
                for embeddings in query_documents_embeddings
            ]
            documents_embeddings = torch.nn.utils.rnn.pad_sequence(
                documents_embeddings, batch_first=True, padding_value=0
            )
            query_scores = colbert_score.colbert_score(
                query.unsqueeze(0), documents_embeddings
            )[0]
            reranked_query_scores, sorted_indices = torch.sort(
                query_scores, descending=True
            )

            # Reorder doc_ids based on the scores
            reranked_query_doc_ids = [
                query_doc_ids[idx] for idx in sorted_indices.tolist()
            ]
            # TODO: create the return during reordering
            res.append(
                [
                    {"id": doc_id, "similarity": score.item()}
                    for doc_id, score in zip(
                        reranked_query_doc_ids, reranked_query_scores
                    )
                ]
            )
            reranked_doc_ids.append(reranked_query_doc_ids)
            reranked_scores.append(reranked_query_scores.cpu().tolist())
        return res
