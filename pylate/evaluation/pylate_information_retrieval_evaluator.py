from __future__ import annotations

import heapq
import json
import logging
import os
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
    This class evaluates an Information Retrieval (IR) setting. This is a direct extension of the InformationRetrievalEvaluator from the sentence-transformers library, only override the compute_metrices method to be compilatible with PyLate models (define asymmetric encoding using is_query params and add padding).
    """

    def __init__(self, *args, truncate_doc_tokens: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncate_doc_tokens = truncate_doc_tokens

    def compute_metrices(
        self,
        model: ColBERT,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
        output_path: str | None = None,
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
        # Compute embedding for the queries
        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            query_embeddings = torch.nn.utils.rnn.pad_sequence(
                model.encode(
                    self.queries,
                    prompt_name=self.query_prompt_name,
                    prompt=self.query_prompt,
                    batch_size=self.batch_size,
                    is_query=True,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=True,
                ),
                batch_first=True,
                padding_value=0,
            )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

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

            # Encode chunk of corpus
            if corpus_embeddings is None:
                with (
                    nullcontext()
                    if self.truncate_dim is None
                    else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
                ):
                    sub_corpus_embeddings = torch.nn.utils.rnn.pad_sequence(
                        corpus_model.encode(
                            self.corpus[corpus_start_idx:corpus_end_idx],
                            prompt_name=self.corpus_prompt_name,
                            prompt=self.corpus_prompt,
                            is_query=False,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            # convert_to_tensor=True,
                            convert_to_numpy=False,
                        ),
                        batch_first=True,
                        padding_value=0,
                    )

            else:
                sub_corpus_embeddings = corpus_embeddings[
                    corpus_start_idx:corpus_end_idx
                ]

            # Truncate document tokens for matryoshka-style evaluation
            if self.truncate_doc_tokens is not None:
                sub_corpus_embeddings = sub_corpus_embeddings[
                    :, : self.truncate_doc_tokens, :
                ]

            # Compute cosine similarities
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
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

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # NOTE: TREC/BEIR/MTEB skips cases where the corpus_id is the same as the query_id, e.g.:
                        # if corpus_id == self.queries_ids[query_itr]:
                        #     continue
                        # This is not done here, as this might be unexpected behaviour if the user just uses
                        # sets of integers from 0 as query_ids and corpus_ids.
                        if len(queries_result_list[name][query_itr]) < max_k:
                            # heaqp tracks the quantity of the first element in the tuple
                            heapq.heappush(
                                queries_result_list[name][query_itr], (score, corpus_id)
                            )
                        else:
                            heapq.heappushpop(
                                queries_result_list[name][query_itr], (score, corpus_id)
                            )

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {
                        "corpus_id": corpus_id,
                        "score": score,
                    }

        if self.write_predictions and output_path is not None:
            for name in queries_result_list:
                base_filename = self.predictions_file.replace(
                    ".jsonl", f"_{name}.jsonl"
                )
                json_path = os.path.join(output_path, base_filename)
                mode = "w"  # Always create a new file for each score function

                with open(json_path, mode=mode, encoding="utf-8") as fOut:
                    for query_itr in range(len(queries_result_list[name])):
                        query_id = self.queries_ids[query_itr]
                        query_text = self.queries[query_itr]
                        results = queries_result_list[name][query_itr]

                        # Sort results by score in descending order
                        results = sorted(
                            results, key=lambda x: x["score"], reverse=True
                        )

                        prediction = {
                            "query_id": query_id,
                            "query": query_text,
                            "results": results,
                        }

                        fOut.write(json.dumps(prediction) + "\n")

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
