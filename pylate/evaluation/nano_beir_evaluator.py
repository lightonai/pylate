from __future__ import annotations

import logging
from typing import Literal

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import is_datasets_available

from .pylate_information_retrieval_evaluator import PyLateInformationRetrievalEvaluator

logger = logging.getLogger(__name__)

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]


dataset_name_to_id = {
    "climatefever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "fiqa2018": "zeta-alpha-ai/NanoFiQA2018",
    "hotpotqa": "zeta-alpha-ai/NanoHotpotQA",
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "nq": "zeta-alpha-ai/NanoNQ",
    "quoraretrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs": "zeta-alpha-ai/NanoSCIDOCS",
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "touche2020": "zeta-alpha-ai/NanoTouche2020",
}

dataset_name_to_human_readable = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class NanoBEIREvaluator(SentenceEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on the NanoBEIR collection of datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
    The datasets are available on HuggingFace at https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6
    The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.


    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import NanoBEIREvaluator

            model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

            datasets = ["QuoraRetrieval", "MSMARCO"]
            query_prompts = {
                "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
                "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: "
            }

            evaluator = NanoBEIREvaluator(
                dataset_names=datasets,
                query_prompts=query_prompts,
            )

            results = evaluator(model)
            '''
            NanoBEIR Evaluation of the model on ['QuoraRetrieval', 'MSMARCO'] dataset:
            Evaluating NanoQuoraRetrieval
            Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
            Queries: 50
            Corpus: 5046

            Score-Function: cosine
            Accuracy@1: 92.00%
            Accuracy@3: 98.00%
            Accuracy@5: 100.00%
            Accuracy@10: 100.00%
            Precision@1: 92.00%
            Precision@3: 40.67%
            Precision@5: 26.00%
            Precision@10: 14.00%
            Recall@1: 81.73%
            Recall@3: 94.20%
            Recall@5: 97.93%
            Recall@10: 100.00%
            MRR@10: 0.9540
            NDCG@10: 0.9597
            MAP@100: 0.9395

            Evaluating NanoMSMARCO
            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
            Queries: 50
            Corpus: 5043

            Score-Function: cosine
            Accuracy@1: 40.00%
            Accuracy@3: 74.00%
            Accuracy@5: 78.00%
            Accuracy@10: 88.00%
            Precision@1: 40.00%
            Precision@3: 24.67%
            Precision@5: 15.60%
            Precision@10: 8.80%
            Recall@1: 40.00%
            Recall@3: 74.00%
            Recall@5: 78.00%
            Recall@10: 88.00%
            MRR@10: 0.5849
            NDCG@10: 0.6572
            MAP@100: 0.5892
            Average Queries: 50.0
            Average Corpus: 5044.5

            Aggregated for Score Function: cosine
            Accuracy@1: 66.00%
            Accuracy@3: 86.00%
            Accuracy@5: 89.00%
            Accuracy@10: 94.00%
            Precision@1: 66.00%
            Recall@1: 60.87%
            Precision@3: 32.67%
            Recall@3: 84.10%
            Precision@5: 20.80%
            Recall@5: 87.97%
            Precision@10: 11.40%
            Recall@10: 94.00%
            MRR@10: 0.7694
            NDCG@10: 0.8085
            '''
            print(evaluator.primary_metric)
            # => "NanoBEIR_mean_cosine_ndcg@10"
            print(results[evaluator.primary_metric])
            # => 0.8084508771660436
    """

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> PyLateInformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the NanoBEIREvaluator."
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(
                dataset_name, None
            )
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(
                dataset_name, None
            )
        human_readable_name = self._get_human_readable_name(dataset_name)
        return PyLateInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )
