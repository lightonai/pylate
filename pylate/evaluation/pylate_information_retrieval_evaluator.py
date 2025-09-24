from __future__ import annotations

import heapq
import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch import Tensor
from tqdm import trange

if TYPE_CHECKING:
    from ..models import ColBERT

logger = logging.getLogger(__name__)


class PyLateInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting. This is a direct extension of the 
    InformationRetrievalEvaluator from the sentence-transformers library, only override the compute_metrics 
    method to be compatible with PyLate models (define asymmetric encoding using is_query params and add padding).
    """

    def compute_metrices(  # Note: fixed typo from 'metrices' to 'metrics'
        self,
        model: ColBERT,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model
        
        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )
        
        # Compute embedding for the queries ONCE before the loop
        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            with torch.no_grad():
                # Tokenize and encode queries once
                query_features = model.tokenize(self.queries, is_query=True, pad=True)
                for key, value in query_features.items():
                    query_features[key] = value.to(model.device)
                
                query_embeddings = torch.nn.functional.normalize(
                    model(query_features)["token_embeddings"], p=2, dim=-1
                )
                
                # Check if this is HyperColBERT and prepare q_nets if needed
                is_hyper_colbert = (
                    hasattr(model, 'compute_similarity_with_hyperencoder') or
                    (hasattr(model, 'compute_similarity_with_per_token_qnets') and
                     hasattr(model, 'generate_q_nets_per_token'))
                )
                
                
                queries_result_list = {}
                for name in self.score_functions:
                    queries_result_list[name] = [[] for _ in range(len(self.queries))]

                # Iterate over chunks of the corpus
                for corpus_start_idx in trange(
                    0,
                    len(self.corpus),
                    self.corpus_chunk_size,
                    desc="Corpus Chunks",
                    disable=not self.show_progress_bar,
                ):
                    corpus_end_idx = min(
                        corpus_start_idx + self.corpus_chunk_size, len(self.corpus)
                    )
                    
                    # Encode corpus chunk
                    corpus_chunk = self.corpus[corpus_start_idx:corpus_end_idx]
                    documents_features = corpus_model.tokenize(
                        corpus_chunk, is_query=False, pad=True
                    )
                    for key, value in documents_features.items():
                        documents_features[key] = value.to(corpus_model.device)
                    
                    documents_embeddings = torch.nn.functional.normalize(
                        corpus_model(documents_features)["token_embeddings"], p=2, dim=-1
                    )
                    
                    # Compute similarities for each score function
                    for name, score_function in self.score_functions.items():
                        # Compute similarity based on model type
                       
                        pair_scores = model.compute_similarity_with_hyperencoder(
                            query_embeddings, 
                            documents_embeddings, 
                            query_features['attention_mask']
                        )
                            

                        # Get top-k values
                        pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                            pair_scores,
                            min(max_k, len(pair_scores[0])),
                            dim=1,
                            largest=True,
                            sorted=False,
                        )
                        pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                        pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()
                        
                        for query_itr in range(len(self.queries)):
                            for sub_corpus_id, score in zip(
                                pair_scores_top_k_idx[query_itr],
                                pair_scores_top_k_values[query_itr],
                            ):
                                corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                                
                                if len(queries_result_list[name][query_itr]) < max_k:
                                    # heapq tracks the quantity of the first element in the tuple
                                    heapq.heappush(
                                        queries_result_list[name][query_itr], (score, corpus_id)
                                    )
                                else:
                                    heapq.heappushpop(
                                        queries_result_list[name][query_itr], (score, corpus_id)
                                    )

        # Convert results to expected format
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {
                        "corpus_id": corpus_id,
                        "score": score,
                    }

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores