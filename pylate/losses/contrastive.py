from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import ColBERT
from ..scores import ColBERTScores
from ..utils import all_gather, all_gather_with_gradients, get_rank, get_world_size


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
        Contrastive scoring callable. Receives queries ``(Q_query, Qt, H)`` and
        stacked documents ``(Q_doc, N, Dt, H)`` and returns ``(Q_query, Q_doc*N)``
        with query-major ordering. Defaults to a :class:`~pylate.scores.ColBERTScores`
        instance.
    score_mini_batch_size
        If set, queries are processed in chunks of this size during scoring.
        Gradients are still computed from a single backward at the end; chunking
        here only reduces transient memory during the forward. Defaults to None
        (no chunking).
    size_average
        Average by the size of the mini-batch.
    gather_across_devices
        Whether to gather the embeddings across devices to have more in batch negatives. We recommend making sure the sampling across GPUs use the same dataset in case of multi-dataset training to make sure the negatives are plausible.

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
        score_metric=None,
        score_mini_batch_size: int | None = None,
        size_average: bool = True,
        gather_across_devices: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super(Contrastive, self).__init__()
        self.score_metric = (
            score_metric if score_metric is not None else ColBERTScores()
        )
        self.model = model
        self.score_mini_batch_size = score_mini_batch_size
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.temperature = temperature

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
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        batch_size = embeddings[0].size(0)
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
        # Note: the queries mask is not used by default; if it's fed through, take care that
        # expansion tokens are not masked from scoring (they may be masked during encoding).
        N = len(embeddings) - 1
        docs_stacked = torch.stack(embeddings[1:], dim=1)
        docs_mask_stacked = torch.stack(masks[1:], dim=1)
        q_mask = masks[0] if not do_query_expansion else None

        # Chunking queries here only lowers transient forward memory — gradients
        # still flow through one big backward below.
        # If the score_mini_batch_size is not set, we process the entire batch at once (old behavior)
        step = self.score_mini_batch_size or batch_size
        score_chunks = []
        for begin in range(0, batch_size, step):
            end = begin + step
            score_chunks.append(
                self.score_metric(
                    embeddings[0][begin:end],
                    docs_stacked,
                    queries_mask=q_mask[begin:end] if q_mask is not None else None,
                    documents_mask=docs_mask_stacked,
                )
            )
        scores = torch.cat(score_chunks, dim=0)

        # Query-major layout: positive for query i is at column i*N.
        labels = torch.arange(batch_size, device=embeddings[0].device) * N
        if self.gather_across_devices:
            labels = labels + get_rank() * batch_size * N

        loss = F.cross_entropy(
            input=scores / self.temperature,
            target=labels,
            reduction="mean" if self.size_average else "sum",
        )

        if self.gather_across_devices:
            loss *= get_world_size()
        return loss
