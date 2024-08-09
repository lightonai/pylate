from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import ColBERT
from ..scores import colbert_scores


def extract_skiplist_mask(
    sentence_features: Iterable[dict[str, torch.Tensor]],
    skiplist: list[int],
) -> list[torch.Tensor]:
    """Extracts the attention masks from the sentence features. We apply a skiplist mask to the documents.
    We skip the first sentence feature because it is the query.

    Examples
    --------
    >>> import torch

    >>> sentence_features = [
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ... ]

    >>> extract_skiplist_mask(
    ...     sentence_features=sentence_features,
    ...     skiplist=[1, 2, 3],
    ... )
    [tensor([[True, True, True, True]]), tensor([[False, False, False,  True]]), tensor([[False, False, False,  True]])]

    """
    attention_masks = [
        sentence_feature["attention_mask"] for sentence_feature in sentence_features
    ]

    skiplist_masks = [
        torch.ones_like(sentence_features[0]["input_ids"], dtype=torch.bool)
    ]

    # We skip the first sentence feature because it is the query.
    skiplist_masks.extend(
        [
            ColBERT.skiplist_mask(
                input_ids=sentence_feature["input_ids"], skiplist=skiplist
            )
            for sentence_feature in sentence_features[1:]
        ]
    )

    return [
        torch.logical_and(skiplist_mask, attention_mask)
        for skiplist_mask, attention_mask in zip(skiplist_masks, attention_masks)
    ]


class Contrastive(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Parameters
    ----------
    model
        ColBERT model.
    score_metric
        ColBERT scoring function. Defaults to colbert_scores.
    size_average
        Average by the size of the mini-batch.

    Examples
    --------
    >>> from giga_cherche import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = losses.Contrastive(model=model)

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

    >>> loss = loss(sentence_features=sentence_features)
    >>> assert isinstance(loss.item(), float)

    """

    def __init__(
        self,
        model: ColBERT,
        score_metric=colbert_scores,
        size_average: bool = True,
    ) -> None:
        super(Contrastive, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.size_average = size_average

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Constrastive loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.

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

        # Note: the queries mask is not used, if added, take care that the expansion tokens are not masked from scoring (because they might be masked during encoding).
        # We might not need to compute the mask for queries but I let the logic there for now
        scores = torch.cat(
            [
                self.score_metric(embeddings[0], group_embeddings, mask)
                for group_embeddings, mask in zip(embeddings[1:], masks[1:])
            ],
            dim=1,
        )

        # create corresponding labels
        # labels = torch.arange(0, rep_anchor.size(0), device=rep_anchor.device)
        labels = torch.arange(0, embeddings[0].size(0), device=embeddings[0].device)
        # compute constrastive loss using cross-entropy over the scores

        return F.cross_entropy(
            input=scores,
            target=labels,
            reduction="mean" if self.size_average else "sum",
        )
