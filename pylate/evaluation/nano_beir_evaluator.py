from __future__ import annotations

import logging
from typing import Literal

from sentence_transformers.evaluation.NanoBEIREvaluator import (
    NanoBEIREvaluator as NanoBEIREvaluatorST,
)
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


MAPPING_DATASET_NAME_TO_ID = {
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

MAPPING_DATASET_NAME_TO_HUMAN_READABLE = {
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


class NanoBEIREvaluator(NanoBEIREvaluatorST):
    """
    This class evaluates the performance of a PyLate Model on the NanoBEIR collection of datasets. This is a direct extension of the NanoBEIREvaluator from the sentence-transformers library, leveraging the PyLateInformationRetrievalEvaluator class.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
    The datasets are available on HuggingFace at https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6
    The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.
    Examples
    --------
    >>> from pylate import models, evaluation
    >>> model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0")
    >>> datasets = ["SciFact"]
    >>> evaluator = evaluation.NanoBEIREvaluator(dataset_names=datasets)
    >>> results = evaluator(model)
    >>> results
    {'NanoSciFact_MaxSim_accuracy@1': 0.62, 'NanoSciFact_MaxSim_accuracy@3': 0.74, 'NanoSciFact_MaxSim_accuracy@5': 0.8, 'NanoSciFact_MaxSim_accuracy@10': 0.86, 'NanoSciFact_MaxSim_precision@1': 0.62, 'NanoSciFact_MaxSim_precision@3': 0.26666666666666666, 'NanoSciFact_MaxSim_precision@5': 0.18, 'NanoSciFact_MaxSim_precision@10': 0.096, 'NanoSciFact_MaxSim_recall@1': 0.595, 'NanoSciFact_MaxSim_recall@3': 0.715, 'NanoSciFact_MaxSim_recall@5': 0.79, 'NanoSciFact_MaxSim_recall@10': 0.85, 'NanoSciFact_MaxSim_ndcg@10': 0.7279903941189909, 'NanoSciFact_MaxSim_mrr@10': 0.6912222222222222, 'NanoSciFact_MaxSim_map@100': 0.6903374780806633, 'NanoBEIR_mean_MaxSim_accuracy@1': 0.62, 'NanoBEIR_mean_MaxSim_accuracy@3': 0.74, 'NanoBEIR_mean_MaxSim_accuracy@5': 0.8, 'NanoBEIR_mean_MaxSim_accuracy@10': 0.86, 'NanoBEIR_mean_MaxSim_precision@1': 0.62, 'NanoBEIR_mean_MaxSim_precision@3': 0.26666666666666666, 'NanoBEIR_mean_MaxSim_precision@5': 0.18, 'NanoBEIR_mean_MaxSim_precision@10': 0.096, 'NanoBEIR_mean_MaxSim_recall@1': 0.595, 'NanoBEIR_mean_MaxSim_recall@3': 0.715, 'NanoBEIR_mean_MaxSim_recall@5': 0.79, 'NanoBEIR_mean_MaxSim_recall@10': 0.85, 'NanoBEIR_mean_MaxSim_ndcg@10': 0.7279903941189909, 'NanoBEIR_mean_MaxSim_mrr@10': 0.6912222222222222, 'NanoBEIR_mean_MaxSim_map@100': 0.6903374780806633}
    """

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> PyLateInformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the NanoBEIREvaluator."
            )
        from datasets import load_dataset

        dataset_path = MAPPING_DATASET_NAME_TO_ID[dataset_name.lower()]
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
