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
    score_metric
        Function that returns a score between two sequences of embeddings.
    size_average
        Average by the size of the mini-batch or perform sum.

    Examples
    --------
    >>> from giga_cherche import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> distillation = losses.Distillation(model=model)

    >>> query = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> documents = model.tokenize([
    ...     "fruits are good for health.",
    ...     "fruits are bad for health."
    ... ], is_query=False)

    >>> sentence_features = [query, documents]

    >>> labels = torch.tensor([
    ...     [0.7, 0.3],
    ... ], dtype=torch.float32)

    >>> loss = distillation(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool = True,
    ) -> None:
        super(Distillation, self).__init__()
        self.score_metric = score_metric
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
            List of tokenized sentences. The first sentence is the query and the rest are documents.
        labels
            The logits for the distillation loss.

        """
        queries_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1
        )
        # Compute the bs * n_ways embeddings
        documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )

        # Reshape them to (bs, n_ways)
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )

        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=self.model.skiplist
        )

        documents_embeddings_mask = masks[1].view(
            queries_embeddings.size(0), -1, *masks[1].shape[1:]
        )
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            documents_embeddings_mask,
        )
        if self.normalize_scores:
            # Compute max and min along the num_scores dimension (dim=1)
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)

            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8

            # Normalize the scores
            scores = (scores - min_scores) / (max_scores - min_scores + epsilon)
        return self.loss_function(
            torch.nn.functional.log_softmax(scores, dim=-1),
            torch.nn.functional.log_softmax(labels, dim=-1),
        )
