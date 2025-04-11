from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import ColBERT
from ..scores import colbert_scores
from ..utils import all_gather, all_gather_with_gradients, get_rank


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
    gather_across_devices
        Whether to gather the embeddings across devices to have more in batch negatives. We recommand making sure the sampling across GPUs use the same dataset in case of multi-dataset training to make sure the negatives are plausible.

    Examples
    --------
    >>> from pylate import models, losses

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
        gather_across_devices: bool = False,
    ) -> None:
        super(Contrastive, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices

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
        labels
            The labels for the contrastive loss. Not used in this implementation, but kept for compatibility with Trainer.

        """
        embeddings = [
            torch.nn.functional.normalize(
                self.model(sentence_feature)["token_embeddings"], p=2, dim=-1
            )
            for sentence_feature in sentence_features
        ]
        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        batch_size = embeddings[0].size(0)
        # create corresponding labels
        labels = torch.arange(0, batch_size, device=embeddings[0].device)
        # Possibly gather the embeddings across devices to have more in-batch negatives.
        if self.gather_across_devices:
            # Note that we only gather the documents embeddings and not the queries embeddings (embeddings[0]), but are keeping gradients. This is to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616
            embeddings = [
                embeddings[0],
                *[
                    torch.cat(all_gather_with_gradients(embedding))
                    for embedding in embeddings[1:]
                ],
            ]
            # Masks [0] is the anchor mask so we do not need to gather it (even though we are not using it for now anyways)
            # Also, we do gather without gradients for the masks as we do not backpropagate through them
            masks = [
                masks[0],
                *[torch.cat(all_gather(mask)) for mask in masks[1:]],
            ]
            rank = get_rank()
            # Adjust the labels to match the gathered embeddings positions
            labels = labels + rank * batch_size
        # Note: the queries mask is not used, if added, take care that the expansion tokens are not masked from scoring (because they might be masked during encoding).
        # We might not need to compute the mask for queries but I let the logic there for now
        scores = torch.cat(
            [
                self.score_metric(embeddings[0], group_embeddings, mask)
                for group_embeddings, mask in zip(embeddings[1:], masks[1:])
            ],
            dim=1,
        )

        # compute constrastive loss using cross-entropy over the scores

        return F.cross_entropy(
            input=scores,
            target=labels,
            reduction="mean" if self.size_average else "sum",
        )
