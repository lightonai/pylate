from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch import Tensor

if TYPE_CHECKING:
    from ..models import ColBERT


logger = logging.getLogger(__name__)


class PlaidInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """This class evaluates an Information Retrieval (IR) setting using a PLAID index.

    It builds an index on the corpus and then queries this index to find the top-k
    most similar documents.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_metrices(
        self,
        model: ColBERT,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
        output_path: str | None = None,
    ) -> dict[str, float]:
        from ..indexes import PLAID

        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Encode the corpus
        logger.info("Encoding corpus...")
        corpus_embeddings = corpus_model.encode(
            self.corpus,
            prompt_name=self.corpus_prompt_name,
            prompt=self.corpus_prompt,
            batch_size=self.batch_size,
            is_query=False,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=False,
            convert_to_numpy=False,
        )

        # Create and build the PLAID index
        logger.info("Building PLAID index...")
        plaid_index = PLAID(
            index_folder="./plaid_index",
            index_name="temp_eval_index",
            override=True,
            embedding_size=128,
        )
        plaid_index.add_documents(
            documents_ids=self.corpus_ids,
            documents_embeddings=corpus_embeddings,
        )

        # Encode the queries
        logger.info("Encoding queries...")
        query_embeddings = model.encode(
            self.queries,
            prompt_name=self.query_prompt_name,
            prompt=self.query_prompt,
            batch_size=self.batch_size,
            is_query=True,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        # Search the index
        logger.info("Searching index...")
        search_results = plaid_index(query_embeddings, k=max_k)

        # Format results
        queries_result_list = {"MaxSim": []}
        for result in search_results:
            formatted_result = [
                {"corpus_id": r["id"], "score": r["score"]} for r in result
            ]
            queries_result_list["MaxSim"].append(formatted_result)

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

        # Compute and output scores
        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in queries_result_list
        }

        for name in scores:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores
