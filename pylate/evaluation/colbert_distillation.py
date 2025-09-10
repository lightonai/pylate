from __future__ import annotations

import itertools
import logging
import os
from contextlib import nullcontext

import torch
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer

from ..scores import colbert_kd_scores
from .colbert_triplet import csv_writer, evaluation_message

logger = logging.getLogger(__name__)


class ColBERTDistillationEvaluator(SentenceEvaluator):
    """ColBERT Distillation Evaluator.
    This class is used to monitor the distillation process of a ColBERT model.

    Parameters
    ----------
    queries
        Set of queries.
    documents
        Set of documents. Each query has a list of documents. Each document is a list of strings.
        Number of documents should be the same for each query.
    scores
        The scores associated with the documents. Each query / documents pairs has a list of scores.
    name
        The name of the evaluator.
    batch_size
        The batch size.
    show_progress_bar
        Whether to show the progress bar.
    write_csv
        Whether to write the results to a CSV file.
    truncate_dim
        The dimension to truncate the embeddings.


    Examples
    --------

    >>> from pylate import models, evaluation

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> queries = [
    ...     "query A",
    ...     "query B",
    ... ]

    >>> documents = [
    ...     ["document A", "document B", "document C"],
    ...     ["document C C", "document B B", "document A A"],
    ... ]

    >>> scores = [
    ...     [0.9, 0.1, 0.05],
    ...     [0.05, 0.9, 0.1],
    ... ]

    >>> distillation_evaluator = evaluation.ColBERTDistillationEvaluator(
    ...     queries=queries,
    ...     documents=documents,
    ...     scores=scores,
    ...     write_csv=True,
    ... )

    >>> results = distillation_evaluator(model=model, output_path=".")


    >>> assert "kl_divergence" in results
    >>> assert isinstance(results["kl_divergence"], float)

    >>> import pandas as pd
    >>> df = pd.read_csv(distillation_evaluator.csv_file)
    >>> assert df.columns.tolist() == distillation_evaluator.csv_headers

    """

    def __init__(
        self,
        queries: list[str],
        documents: list[list[str]],
        scores: list[list[float]],
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        normalize_scores: bool = True,
    ) -> None:
        super().__init__()
        assert len(queries) == len(documents)
        self.queries = queries
        # Flatten the documents list
        self.documents = list(itertools.chain.from_iterable(documents))
        self.scores = scores
        self.name = name
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.normalize_scores = normalize_scores
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        self.csv_file: str = (
            "distillation_evaluation" + ("_" + name if name else "") + "_results.csv"
        )

        self.csv_headers = [
            "epoch",
            "steps",
            "kl_divergence",
        ]

    def __call__(
        self,
        model: "SentenceTransformer",
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        evaluation_message(
            epoch=epoch, steps=steps, name=self.name, truncate_dim=self.truncate_dim
        )

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            queries_embeddings = torch.nn.utils.rnn.pad_sequence(
                model.encode(
                    self.queries,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=True,
                    is_query=True,
                ),
                batch_first=True,
                padding_value=0,
            )

            documents_embeddings = torch.nn.utils.rnn.pad_sequence(
                model.encode(
                    self.documents,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=False,
                    is_query=False,
                ),
                batch_first=True,
                padding_value=0,
            )
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )
        scores = colbert_kd_scores(
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )
        if self.normalize_scores:
            # Compute max and min along the num_scores dimension (dim=1)
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)

            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8

            # Normalize the scores
            scores = (scores - min_scores) / (max_scores - min_scores + epsilon)
        kl_divergence = self.loss(
            torch.nn.functional.log_softmax(scores, dim=-1),
            torch.nn.functional.log_softmax(
                torch.tensor(self.scores, device=scores.device), dim=-1
            ),
        ).item()
        metrics = self.prefix_name_to_metrics(
            {"kl_divergence": kl_divergence}, self.name
        )
        self.store_metrics_in_model_card_data(model, metrics)

        if output_path is not None and self.write_csv:
            csv_writer(
                path=os.path.join(output_path, self.csv_file),
                data=[epoch, steps, kl_divergence],
                header=self.csv_headers,
            )

        return metrics
