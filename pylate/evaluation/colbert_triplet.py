from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext

from sentence_transformers.evaluation import TripletEvaluator

from ..models import ColBERT
from ..scores import colbert_scores_pairwise

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def csv_writer(
    path: str | None,
    data: list[str | float | int],
    header: list[str],
) -> None:
    """Write the results to a CSV file.

    Parameters
    ----------
    path
        The path to the CSV file.
    rows
        The rows to write to the CSV file.
    header
        The header of the CSV file.

    Examples
    --------
    >>> import pandas as pd

    >>> csv_writer(
    ...     path="results.csv",
    ...     data=[1, 2, 0.8],
    ...     header=["epoch", "steps", "accuracy"],
    ... )

    >>> df = pd.read_csv("results.csv")

    >>> assert df.columns.tolist() == ["epoch", "steps", "accuracy"]

    """
    mode = "w" if not os.path.isfile(path=path) else "a"
    with open(file=path, newline="", mode=mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)
        writer.writerow(data)


def evaluation_message(
    epoch: int,
    steps: int,
    name: str,
    truncate_dim: int | None = None,
) -> None:
    """Prints an evaluation message."""
    out_txt = ""
    if epoch != -1:
        out_txt = (
            f" after epoch {epoch}"
            if steps == -1
            else f" in epoch {epoch} after {steps} steps"
        )

    if truncate_dim is not None:
        out_txt += f" (truncated to {truncate_dim})"

    logger.info(
        f"TripletEvaluator: Evaluating the model on the {name} dataset{out_txt}:"
    )


class ColBERTTripletEvaluator(TripletEvaluator):
    """Evaluate a model based on a set of triples. The evaluation will compare the
    score between the anchor and the positive sample with the score between the anchor
    and the negative sample. The accuracy is computed as the number of times the score
    between the anchor and the positive sample is higher than the score between the anchor
    and the negative sample.

    Parameters
    ----------
    anchors
        Sentences to check similarity to. (e.g. a query)
    positives
        List of positive sentences
    negatives
        List of negative sentences
    name
        Name for the output.
    batch_size
        Batch size used to compute embeddings.
    show_progress_bar
        If true, prints a progress bar.
    write_csv
        Wether or not to write results to a CSV file.
    truncate_dim
        The dimension to truncate sentence embeddings to. If None, do not truncate.

    Examples
    --------
    >>> from pylate import evaluation, models

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     device="cpu",
    ... )

    >>> anchors = [
    ...     "fruits are healthy.",
    ...     "fruits are healthy.",
    ... ]

    >>> positives = [
    ...     "fruits are good for health.",
    ...     "Fruits are growing in the trees.",
    ... ]

    >>> negatives = [
    ...     "Fruits are growing in the trees.",
    ...     "fruits are good for health.",
    ... ]

    >>> triplet_evaluation = evaluation.ColBERTTripletEvaluator(
    ...     anchors=anchors,
    ...     positives=positives,
    ...     negatives=negatives,
    ...     write_csv=True,
    ... )

    >>> results = triplet_evaluation(model=model, output_path=".")

    >>> results
    {'accuracy': 0.5}

    >>> triplet_evaluation.csv_headers
    ['epoch', 'steps', 'accuracy']

    >>> import pandas as pd
    >>> df = pd.read_csv(triplet_evaluation.csv_file)
    >>> assert df.columns.tolist() == triplet_evaluation.csv_headers

    """

    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ) -> None:
        super().__init__(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_distance_function=None,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
        )

        self.csv_headers = [
            "epoch",
            "steps",
            "accuracy",
        ]

        self.metrics = [
            "accuracy",
        ]

        self.primary_metric = "accuracy"

    def __call__(
        self,
        model: ColBERT,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        """Evaluate the model on the triplet dataset. Measure the scoring between the anchor
        and the positive with every other positive and negative samples using HITS@K.
        """
        evaluation_message(
            epoch=epoch, steps=steps, name=self.name, truncate_dim=self.truncate_dim
        )

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(truncate_dim=self.truncate_dim)
        ):
            embeddings_anchors = model.encode(
                sentences=self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                is_query=True,
            )
            embeddings_positives = model.encode(
                sentences=self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )
            embeddings_negatives = model.encode(
                sentences=self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )

        # Colbert pairwise scores
        positive_scores = colbert_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_positives,
        )

        negative_scores = colbert_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_negatives,
        )

        metrics = {
            "accuracy": (
                sum(positive_scores > negative_scores) / len(positive_scores)
            ).item()
        }

        for metric in self.metrics:
            logger.info(f"{metric.capitalize()}: \t{metrics[metric]:.2f}")

        self.store_metrics_in_model_card_data(model=model, metrics=metrics)

        if output_path is not None and self.write_csv:
            csv_writer(
                path=os.path.join(output_path, self.csv_file),
                header=self.csv_headers,
                data=[
                    epoch,
                    steps,
                ]
                + [metrics[metric] for metric in self.metrics],
            )

        return metrics
