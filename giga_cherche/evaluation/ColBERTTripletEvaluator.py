import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from giga_cherche.util import colbert_pairwise_score, colbert_score
import numpy as np

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction

import torch

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class ColBERTTripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
    Checks if colbert distance(sentence, positive_example) < distance(sentence, negative_example).

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import TripletEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load a dataset with (anchor, positive, negative) triplets
            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

            # Initialize the TripletEvaluator using anchors, positives, and negatives
            triplet_evaluator = TripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all-nli-dev",
            )
            results = triplet_evaluator(model)
            '''
            TripletEvaluator: Evaluating the model on the all-nli-dev dataset:
            Accuracy Cosine Distance:        95.60
            Accuracy Dot Product:            4.40
            Accuracy Manhattan Distance:     95.40
            Accuracy Euclidean Distance:     95.60
            '''
            print(triplet_evaluator.primary_metric)
            # => "all-nli-dev_max_accuracy"
            print(results[triplet_evaluator.primary_metric])
            # => 0.956
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        main_distance_function: Optional[Union[str, SimilarityFunction]] = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        """
        Initializes a TripletEvaluator object.

        Args:
            anchors (List[str]): Sentences to check similarity to. (e.g. a query)
            positives (List[str]): List of positive sentences
            negatives (List[str]): List of negative sentences
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
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name
        self.truncate_dim = truncate_dim

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = SimilarityFunction(main_distance_function) if main_distance_function else None

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)
    #TODO: add mAP and other metrics
    def __call__(
        self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        num_triplets = 0
        (
            num_correct_colbert_triplets
        ) = 0

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings_anchors = model.encode(
                self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                is_query=True,
            )
            embeddings_positives = model.encode(
                self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )
            embeddings_negatives = model.encode(
                self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                is_query=False,
            )

        # TODO: do the padding in encode()?
        embeddings_positives = torch.nn.utils.rnn.pad_sequence(embeddings_positives, batch_first=True, padding_value=0)
        attention_mask_positives = (embeddings_positives.sum(dim=-1) != 0).float()
   
        embeddings_negatives = torch.nn.utils.rnn.pad_sequence(embeddings_negatives, batch_first=True, padding_value=0)
        attention_mask_negatives = (embeddings_negatives.sum(dim=-1) != 0).float()

        # Colbert distance
        # pos_colbert_distances = colbert_pairwise_score(embeddings_anchors, embeddings_positives)
        # neg_colbert_distances = colbert_pairwise_score(embeddings_anchors, embeddings_negatives)
        pos_colbert_distances_full = colbert_score(embeddings_anchors, embeddings_positives, attention_mask_positives)
        neg_colbert_distances_full = colbert_score(embeddings_anchors, embeddings_negatives, attention_mask_negatives)
        distances_full = torch.cat([pos_colbert_distances_full, neg_colbert_distances_full], dim=1)
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
            if pos_colbert_distances[idx] < neg_colbert_distances[idx]:
                num_correct_colbert_triplets += 1
        
        accuracy_colbert = num_correct_colbert_triplets / num_triplets

        logger.info("Accuracy Colbert:   \t{:.2f}".format(accuracy_colbert * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_colbert])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_colbert])

        self.primary_metric = {
            SimilarityFunction.COSINE: "cosine_accuracy",
            SimilarityFunction.DOT_PRODUCT: "dot_accuracy",
            SimilarityFunction.EUCLIDEAN: "euclidean_accuracy",
            SimilarityFunction.MANHATTAN: "manhattan_accuracy",
        }.get(self.main_distance_function, "max_accuracy")
        metrics = {
            "colbert_accuracy": accuracy_colbert,
            "hits@1": np.sum(ranks <= 1)/len(ranks),
            "hits@5": np.sum(ranks <= 5)/len(ranks),
            "hits@10": np.sum(ranks <= 10)/len(ranks),
            "hits@25": np.sum(ranks <= 25)/len(ranks),
            # "max_accuracy": max(accuracy_cos, accuracy_manhattan, accuracy_euclidean),
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics
