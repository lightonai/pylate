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
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class PyLateInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    Example:
        ::

            import random
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import InformationRetrievalEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)
            corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
            queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
            relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

            # For this dataset, we want to concatenate the title and texts for the corpus
            corpus = corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])

            # Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
            required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
            required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
            corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

            # Convert the datasets to dictionaries
            corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
            queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
            relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
                qid = str(qid)
                corpus_ids = str(corpus_ids)
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                relevant_docs[qid].add(corpus_ids)

            # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="BeIR-touche2020-subset-test",
            )
            results = ir_evaluator(model)
            '''
            Information Retrieval Evaluation of the model on the BeIR-touche2020-test dataset:
            Queries: 49
            Corpus: 31923

            Score-Function: cosine
            Accuracy@1: 77.55%
            Accuracy@3: 93.88%
            Accuracy@5: 97.96%
            Accuracy@10: 100.00%
            Precision@1: 77.55%
            Precision@3: 72.11%
            Precision@5: 71.43%
            Precision@10: 62.65%
            Recall@1: 1.72%
            Recall@3: 4.78%
            Recall@5: 7.90%
            Recall@10: 13.86%
            MRR@10: 0.8580
            NDCG@10: 0.6606
            MAP@100: 0.2934
            '''
            print(ir_evaluator.primary_metric)
            # => "BeIR-touche2020-test_cosine_map@100"
            print(results[ir_evaluator.primary_metric])
            # => 0.29335196224364596
    """

    def compute_metrices(
        self,
        model: SentenceTransformer,
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
        # Compute embedding for the queries
        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            query_embeddings = model.encode(
                self.queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                is_query=True,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
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

            # Compute cosine similarites
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
