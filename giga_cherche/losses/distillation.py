from typing import Callable, Iterable

import torch

from ..models import ColBERT
from ..scores import colbert_kd_scores
from .contrastive import extract_skiplist_mask


class Distillation(torch.nn.Module):
    """Distillation loss for ColBERT model. The loss is computed with respect to the format of SentenceTransformer library.

    Parameters
    ----------
    model
        SentenceTransformer model.
    distance_metric
        Function that returns a distance between two embeddings.
    size_average
        Average by the size of the mini-batch or perform sum.

    Examples
    --------
    >>> from giga_cherche import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> distillation = losses.Distillation(model=model)

    >>> anchor = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> positive = model.tokenize([
    ...     "fruits are good for health.",
    ... ], is_query=False)

    >>> negative = model.tokenize([
    ...     "fruits are bad for health.",
    ... ], is_query=False)

    >>> sentence_features = [anchor, positive, negative]

    >>> labels = torch.tensor([
    ...     [0.7, 0.3],
    ... ], dtype=torch.float32)

    >>> loss = distillation(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        distance_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool = True,
    ) -> None:
        super(Distillation, self).__init__()
        self.distance_metric = distance_metric
        self.model = model
        self.loss_function = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum", log_target=True
        )
        self.normalize_scores = normalize_scores

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the distillation loss with respect to SentenceTransformer.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.
        labels
            The logits for the distillation loss.

        """
        embeddings = [
            torch.nn.functional.normalize(
                self.model(sentence_feature)["token_embeddings"], p=2, dim=-1
            )
            for sentence_feature in sentence_features
        ]

        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=self.model.skiplist
        )

        # Compute the distance between the anchor and positive/negative embeddings.
        anchor_embeddings = embeddings[0]
        positive_negative_embeddings = torch.stack(embeddings[1:], dim=1)
        positive_negative_embeddings_mask = torch.stack(masks[1:], dim=1)

        distances = self.distance_metric(
            anchor_embeddings,
            positive_negative_embeddings,
            positive_negative_embeddings_mask,
        )
        if self.normalize_scores:
            # Compute max and min along the num_scores dimension (dim=1)
            max_distances, _ = torch.max(distances, dim=1, keepdim=True)
            min_distances, _ = torch.min(distances, dim=1, keepdim=True)

            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8

            # Normalize the scores
            distances = (distances - min_distances) / (
                max_distances - min_distances + epsilon
            )
        return self.loss_function(
            torch.nn.functional.log_softmax(distances, dim=-1),
            torch.nn.functional.log_softmax(labels, dim=-1),
        )
