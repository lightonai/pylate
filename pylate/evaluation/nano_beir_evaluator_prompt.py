from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
from sentence_transformers import SimilarityFunction
from sentence_transformers.evaluation.NanoBEIREvaluator import (
    NanoBEIREvaluator as NanoBEIREvaluatorST,
)
from sentence_transformers.util import is_datasets_available
from torch import Tensor

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

MAPPING_DATASET_NAME_TO_PROMPT = {
    "climatefever": "A relevant document should also provide a clear and concise explanation, avoiding unnecessary complexity or ambiguity. When in doubt, prioritize documents that provide a clear, direct, and specific answer to the query.",
    "dbpedia": "A document that meets these criteria is considered relevant, while a document that does not meet these criteria is considered non-relevant.",
    "fever": "Think carefully about relevance",
    "fiqa2018": "A document that meets these criteria is considered relevant, while a document that does not meet these criteria is considered non-relevant",
    "hotpotqa": "Think carefully about relevance.",
    "msmarco": "A document that meets these criteria is considered relevant, while a document that does not meet these criteria is considered non-relevant.",
    "nfcorpus": "A document that meets these criteria is considered relevant, while a document that does not meet these criteria is considered non-relevant.",
    "nq": "A document that meets these criteria is considered relevant, while a document that does not meet these criteria is considered non-relevant.",
    "quoraretrieval": "A relevant document should focus solely on providing a clear and accurate answer to the query, without distracting or unnecessary information",
    "scidocs": "A relevant document should also provide a clear and concise explanation, avoiding unnecessary complexity or ambiguity. When in doubt, prioritize documents that provide a clear, direct, and specific answer to the query.",
    "arguana": "A relevant document should also provide a clear and concise explanation, avoiding unnecessary complexity or ambiguity. When in doubt, prioritize documents that provide a clear, direct, and specific answer to the query.",
    "scifact": "Think carefully about these conditions when determining relevance.",
    "touche2020": "Think carefully about relevance",
}


class NanoBEIREvaluatorWithPrompt(NanoBEIREvaluatorST):
    """Evaluate the performance of a PyLate Model on the NanoBEIR collection.


    This is a direct extension of the NanoBEIREvaluator from the
    sentence-transformers library, leveraging the
    PyLateInformationRetrievalEvaluator class. The collection is a set of datasets
    based on the BEIR collection, but with a significantly smaller size, so it
    can be used for quickly evaluating the retrieval performance of a
    model before commiting to a full evaluation.
    The Evaluator will return the same metrics as the InformationRetrievalEvaluator
    (i.e., MRR, nDCG, Recall@k), for each dataset and on average.

    Examples
    --------
    >>> from pylate import models, evaluation

    >>> model = models.ColBERT(
    ...     model_name_or_path="lightonai/colbertv2.0"
    ... )

    >>> datasets = ["SciFact"]

    >>> try:
    ...     evaluator = evaluation.NanoBEIREvaluator(
    ...         dataset_names=datasets
    ...     )
    ...     results = evaluator(model)
    ... except Exception:
    ...     pass

    {'NanoSciFact_MaxSim_accuracy@1': 0.62, 'NanoSciFact_MaxSim_accuracy@3': 0.74, 'NanoSciFact_MaxSim_accuracy@5': 0.8, 'NanoSciFact_MaxSim_accuracy@10': 0.86, 'NanoSciFact_MaxSim_precision@1': 0.62, 'NanoSciFact_MaxSim_precision@3': 0.26666666666666666, 'NanoSciFact_MaxSim_precision@5': 0.18, 'NanoSciFact_MaxSim_precision@10': 0.096, 'NanoSciFact_MaxSim_recall@1': 0.595, 'NanoSciFact_MaxSim_recall@3': 0.715, 'NanoSciFact_MaxSim_recall@5': 0.79, 'NanoSciFact_MaxSim_recall@10': 0.85, 'NanoSciFact_MaxSim_ndcg@10': 0.7279903941189909, 'NanoSciFact_MaxSim_mrr@10': 0.6912222222222222, 'NanoSciFact_MaxSim_map@100': 0.6903374780806633, 'NanoBEIR_mean_MaxSim_accuracy@1': 0.62, 'NanoBEIR_mean_MaxSim_accuracy@3': 0.74, 'NanoBEIR_mean_MaxSim_accuracy@5': 0.8, 'NanoBEIR_mean_MaxSim_accuracy@10': 0.86, 'NanoBEIR_mean_MaxSim_precision@1': 0.62, 'NanoBEIR_mean_MaxSim_precision@3': 0.26666666666666666, 'NanoBEIR_mean_MaxSim_precision@5': 0.18, 'NanoBEIR_mean_MaxSim_precision@10': 0.096, 'NanoBEIR_mean_MaxSim_recall@1': 0.595, 'NanoBEIR_mean_MaxSim_recall@3': 0.715, 'NanoBEIR_mean_MaxSim_recall@5': 0.79, 'NanoBEIR_mean_MaxSim_recall@10': 0.85, 'NanoBEIR_mean_MaxSim_ndcg@10': 0.7279903941189909, 'NanoBEIR_mean_MaxSim_mrr@10': 0.6912222222222222, 'NanoBEIR_mean_MaxSim_map@100': 0.6903374780806633}

    References
    ----------
    - [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)

    """

    # override init just to reset the name to include "prompt"
    def __init__(
        self,
        dataset_names: list[DatasetNameType] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        write_predictions: bool = False,
    ):
        super().__init__(
            dataset_names=dataset_names,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
            write_predictions=write_predictions,
        )
        self.name = f"NanoBEIRprompt_{aggregate_key}"
        self.information_retrieval_class = PyLateInformationRetrievalEvaluator

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
            + f". {MAPPING_DATASET_NAME_TO_PROMPT.get(dataset_name, '')}"
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
            name=human_readable_name + "prompt",
            corpus_chunk_size=1000,
            **ir_evaluator_kwargs,
        )
