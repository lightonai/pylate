import csv
import logging
import os
from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer

from ..scores import colbert_kd_scores

logger = logging.getLogger(__name__)

__all__ = ["ColBERTDistillationEvaluator"]


class ColBERTDistillationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a query, a list of documents and the associated scores from a teacher.
    """

    def __init__(
        self,
        queries: list[str],
        documents: list[list[str]],
        scores: list[list[float]],
        distance_metric: Callable = colbert_kd_scores,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ) -> None:
        """
        Initializes a TripletEvaluator object.

        Args:
            queries (list[str]): Sentences to check similarity to. (e.g. a query)
            documents (list[list[str]]): List of list of documents to compare to the queries.
            scores (list[float]): List of teacher scores
            main_distance_function (Union[str, SimilarityFunction], optional):
                The distance function to use. If not specified, use cosine similarity,
                dot product, Euclidean, and Manhattan. Defaults to None.
            name (str): Name for the output. Defaults to "".
            batch_size (int): Batch size used to compute embeddings. Defaults to 16.
            show_progress_bar (bool): If true, prints a progress bar. Defaults to False.
            write_csv (bool): Write results to a CSV file. Defaults to True.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                `None` uses the model's current truncation dimension. Defaults to None.
        """
        super().__init__()
        self.queries = queries
        self.documents = documents
        self.scores = scores

        self.distance_metric = distance_metric
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.name = name
        self.truncate_dim = truncate_dim

        for i in range(len(self.documents)):
            assert len(self.queries) == len(self.documents[i])

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = (
            "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "accuracy_cosinus",
            "accuracy_manhattan",
            "accuracy_euclidean",
        ]
        self.write_csv = write_csv

    # TODO: add mAP and other metrics
    def __call__(
        self,
        model: "SentenceTransformer",
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:"
        )

        num_triplets = 0
        (num_correct_colbert_triplets) = 0

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            embeddings_query = torch.stack(
                model.encode(
                    self.queries,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=True,
                    is_query=True,
                )
            )

            embeddings_documents = [
                torch.nn.utils.rnn.pad_sequence(
                    model.encode(
                        documents,
                        batch_size=self.batch_size,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_numpy=False,
                        is_query=False,
                    ),
                    batch_first=True,
                    padding_value=0,
                )
                for documents in self.documents
            ]
            max_length = max([len(documents[0]) for documents in embeddings_documents])
            embeddings_documents = [
                torch.nn.functional.pad(
                    batch,
                    (0, 0, 0, max_length - batch.size(1)),
                    mode="constant",
                    value=0,
                )
                for batch in embeddings_documents
            ]

            # embeddings_documents = torch.nn.utils.rnn.pad_sequence(
            #     embeddings_documents, batch_first=True, padding_value=0
            # )
            embeddings_documents = torch.stack(embeddings_documents, dim=1)

        distances = self.distance_metric(embeddings_query, embeddings_documents)
        KL_div = self.KLDiv(
            torch.nn.functional.log_softmax(distances, dim=-1),
            torch.nn.functional.log_softmax(
                torch.tensor(self.scores, device=distances.device), dim=-1
            ),
        ).item()

        metrics = {"KL_div": KL_div}

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, KL_div])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, KL_div])

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)

        return metrics
        # TODO: we added the option to do the padding in encode, but it returns a list of tensors for consistency, so would require to do list -> tensor -> list -> tensor
        # We do not need masking as padding with zeros vectors yields 0 cosine similarity

        # Colbert distance
        # pos_colbert_distances = colbert_pairwise_score(embeddings_anchors, embeddings_positives)
        # neg_colbert_distances = colbert_pairwise_score(embeddings_anchors, embeddings_negatives)
        pos_colbert_distances_full = colbert_score(
            embeddings_anchors, embeddings_positives
        )
        neg_colbert_distances_full = colbert_score(
            embeddings_anchors, embeddings_negatives
        )
        distances_full = torch.cat(
            [pos_colbert_distances_full, neg_colbert_distances_full], dim=1
        )
        # print(distances_full.shape)
        labels = np.arange(0, len(embeddings_anchors))
        indices = np.argsort(-distances_full.cpu().numpy(), axis=1)
        ranks = indices.argsort()
        # print(ranks.shape)
        ranks = ranks[np.arange(len(labels)), labels]
        ranks = ranks + 1
        pos_colbert_distances = pos_colbert_distances_full.diag()
        neg_colbert_distances = neg_colbert_distances_full.diag()

        for idx in range(len(pos_colbert_distances)):
            num_triplets += 1
            if pos_colbert_distances[idx] > neg_colbert_distances[idx]:
                num_correct_colbert_triplets += 1

        accuracy_colbert = num_correct_colbert_triplets / num_triplets

        logger.info("Accuracy Colbert:   \t{:.2f}".format(accuracy_colbert * 100))
