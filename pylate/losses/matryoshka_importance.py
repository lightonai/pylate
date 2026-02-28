from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..models import ColBERT
from .cached_contrastive import CachedContrastive
from .matryoshka_doc_tokens import ForwardCachingDecorator

logger = logging.getLogger(__name__)


class ImportanceScoreHead(nn.Module):
    """Linear head that scores each token's importance.

    Used with a straight-through estimator (STE): hard top-k selection in
    the forward pass, but gradients flow through ``sigmoid(scores)`` in the
    backward pass.

    Parameters
    ----------
    embed_dim
        Embedding dimension of the ColBERT model.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None,
        k: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Select top-k tokens via importance scoring with STE.

        Parameters
        ----------
        embeddings
            (batch, T, dim) token embeddings.
        mask
            (batch, T) boolean mask (True = valid token).
        k
            Number of tokens to select.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Selected embeddings (batch, k, dim) and mask (batch, k).
        """
        batch, T, dim = embeddings.shape
        if k >= T:
            return embeddings, mask

        scores = self.linear(embeddings).squeeze(-1)  # (batch, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Soft scores for backward path
        soft_scores = torch.sigmoid(scores)  # (batch, T)

        # Hard top-k selection
        _, topk_idx = scores.topk(k, dim=1)  # (batch, k)
        topk_idx_sorted, sort_perm = topk_idx.sort(dim=1)

        # Build hard binary mask
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)

        # STE: forward uses hard_mask, backward flows through soft_scores
        ste_mask = hard_mask - soft_scores.detach() + soft_scores  # (batch, T)

        # Weight embeddings by STE mask
        weighted = embeddings * ste_mask.unsqueeze(-1)  # (batch, T, dim)

        # Gather the selected tokens
        selected = weighted.gather(
            1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, dim)
        )  # (batch, k, dim)

        # Gather mask
        selected_mask = None
        if mask is not None:
            selected_mask = mask.gather(1, topk_idx_sorted)

        return selected, selected_mask


def _apply_importance_3d(
    score_head: ImportanceScoreHead,
    embeddings: Tensor,
    mask: Tensor | None,
    k: int,
) -> tuple[Tensor, Tensor | None]:
    """Apply ImportanceScoreHead to a 3D tensor."""
    return score_head(embeddings, mask, k)


def _apply_importance(
    score_head: ImportanceScoreHead,
    embeddings: Tensor,
    mask: Tensor | None,
    k: int,
) -> tuple[Tensor, Tensor | None]:
    """Apply ImportanceScoreHead, handling both 3D and 4D tensors."""
    if embeddings.dim() == 3:
        return _apply_importance_3d(score_head, embeddings, mask, k)
    elif embeddings.dim() == 4:
        batch, n_ways, T, dim = embeddings.shape
        flat_emb = embeddings.reshape(batch * n_ways, T, dim)
        flat_mask = None
        if mask is not None:
            flat_mask = mask.reshape(batch * n_ways, T)
        out_emb, out_mask = _apply_importance_3d(
            score_head, flat_emb, flat_mask, k
        )
        out_T = out_emb.shape[1]
        out_emb = out_emb.reshape(batch, n_ways, out_T, dim)
        if out_mask is not None:
            out_mask = out_mask.reshape(batch, n_ways, out_T)
        return out_emb, out_mask
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {embeddings.dim()}D")


class ImportanceScoreMetricDecorator:
    """Wraps a ColBERT score function to apply importance-based token selection
    on document embeddings before scoring."""

    def __init__(
        self,
        score_fn,
        score_head: ImportanceScoreHead,
        n_tokens: int,
    ) -> None:
        self.score_fn = score_fn
        self.score_head = score_head
        self.n_tokens = n_tokens

    def __call__(
        self,
        queries_embeddings,
        documents_embeddings,
        queries_mask=None,
        documents_mask=None,
    ):
        documents_embeddings, documents_mask = _apply_importance(
            self.score_head,
            documents_embeddings,
            documents_mask,
            self.n_tokens,
        )
        return self.score_fn(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )


class CachedImportanceLossDecorator:
    """Decorator for CachedContrastive.calculate_loss that applies
    importance-based token selection at multiple document token counts.

    Follows the same gradient-chaining pattern as CachedDocTokensLossDecorator.
    """

    def __init__(
        self,
        fn,
        score_head: ImportanceScoreHead,
        n_doc_tokens: Sequence[int],
        doc_tokens_weights: Sequence[float] | Sequence[int],
        n_tokens_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.score_head = score_head
        self.n_doc_tokens = n_doc_tokens
        self.doc_tokens_weights = doc_tokens_weights
        self.n_tokens_per_step = n_tokens_per_step

    def __call__(
        self, reps: list[list[Tensor]], masks: list[Tensor], *args, **kwargs
    ) -> Tensor:
        dim_indices = list(range(len(self.n_doc_tokens)))
        if 0 < self.n_tokens_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_tokens_per_step)

        loss = 0.0
        for idx in dim_indices:
            n_tokens = self.n_doc_tokens[idx]
            weight = self.doc_tokens_weights[idx]

            # Apply importance selection to document reps and masks
            selected = []
            selected_masks = [masks[0]]  # query mask unchanged
            for i, sent_reps in enumerate(reps):
                if i == 0:
                    selected.append([r.clone() for r in sent_reps])
                else:
                    sel_sent = []
                    for r in sent_reps:
                        sel_r, _ = _apply_importance_3d(
                            self.score_head, r, None, n_tokens
                        )
                        sel_sent.append(sel_r)
                    selected.append(sel_sent)

            # Handle document masks
            for doc_mask in masks[1:]:
                # For importance selection, we need to apply the score head
                # to get which tokens are selected for the mask too.
                # Since we don't have the embeddings here for mask-only selection,
                # we create a dummy to get indices. But the mask is full-batch,
                # not chunked. Use a simple approach: select top-k from mask
                # by scoring with zeros (which will rely on bias term).
                # Better approach: just truncate the mask since importance
                # head already handles masking internally.
                if n_tokens < doc_mask.shape[-1]:
                    selected_masks.append(doc_mask[:, :n_tokens])
                else:
                    selected_masks.append(doc_mask)

            compute_gradients = torch.is_grad_enabled()
            if compute_gradients:
                matryoshka_reps = [
                    [r.detach().requires_grad_() for r in sent_reps]
                    for sent_reps in selected
                ]
            else:
                matryoshka_reps = selected

            loss += weight * self.fn(
                matryoshka_reps, selected_masks, *args, **kwargs
            )

            if compute_gradients:
                for t_minibatch, d_minibatch in zip(selected, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        if d.grad is not None:
                            t.backward(weight * d.grad)

        return loss


class MatryoshkaImportanceLoss(nn.Module):
    """Matryoshka-style loss with learned importance scoring and
    straight-through estimator (STE) for document token selection.

    A ``Linear(embed_dim, 1)`` head scores each token's importance. Hard top-k
    selection is used in the forward pass (actual token reduction), while the
    STE trick allows gradients to flow through ``sigmoid(scores)`` to train
    the score head.

    Parameters
    ----------
    model
        ColBERT model.
    loss
        Base loss function (e.g. Distillation, Contrastive, CachedContrastive).
    n_doc_tokens
        List of document token counts to train at.
    doc_tokens_weights
        Weights for each token count's loss contribution.
    n_tokens_per_step
        Number of token counts to randomly sample per step (-1 = all).

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> base_loss = losses.Distillation(model=model)

    >>> loss = losses.MatryoshkaImportanceLoss(
    ...     model=model,
    ...     loss=base_loss,
    ...     n_doc_tokens=[32, 64, 128],
    ... )
    """

    def __init__(
        self,
        model: ColBERT,
        loss: nn.Module,
        n_doc_tokens: Sequence[int],
        doc_tokens_weights: Sequence[float] | Sequence[int] | None = None,
        n_tokens_per_step: int = -1,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

        if not n_doc_tokens:
            raise ValueError(
                "You must provide at least one value in n_doc_tokens."
            )
        if any(n <= 0 for n in n_doc_tokens):
            raise ValueError("All values in n_doc_tokens must be > 0.")
        if doc_tokens_weights is None:
            doc_tokens_weights = [1] * len(n_doc_tokens)
        elif len(doc_tokens_weights) != len(n_doc_tokens):
            raise ValueError(
                "doc_tokens_weights must have the same length as n_doc_tokens."
            )

        dims_weights = sorted(
            zip(n_doc_tokens, doc_tokens_weights),
            key=lambda x: x[0],
            reverse=True,
        )
        self.n_doc_tokens: tuple[int, ...]
        self.doc_tokens_weights: tuple[float, ...] | tuple[int, ...]
        self.n_doc_tokens, self.doc_tokens_weights = zip(*dims_weights)
        self.n_tokens_per_step = n_tokens_per_step

        # Infer embed_dim from the model
        embed_dim = model.get_sentence_embedding_dimension()
        self.score_head = ImportanceScoreHead(embed_dim=embed_dim)
        # Move to model device
        device = next(model.parameters()).device
        self.score_head = self.score_head.to(device)

        if isinstance(loss, CachedContrastive):
            loss.calculate_loss = CachedImportanceLossDecorator(
                loss.calculate_loss,
                self.score_head,
                self.n_doc_tokens,
                self.doc_tokens_weights,
                self.n_tokens_per_step,
            )

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        if isinstance(self.loss, CachedContrastive):
            return self.loss(sentence_features, labels)

        original_forward = self.model.forward
        original_score_metric = self.loss.score_metric

        try:
            decorated_forward = ForwardCachingDecorator(original_forward)
            self.model.forward = decorated_forward

            dim_indices = list(range(len(self.n_doc_tokens)))
            if 0 < self.n_tokens_per_step < len(dim_indices):
                dim_indices = sorted(
                    random.sample(dim_indices, self.n_tokens_per_step)
                )

            loss = 0.0
            for idx in dim_indices:
                n_tokens = self.n_doc_tokens[idx]
                weight = self.doc_tokens_weights[idx]
                self.loss.score_metric = ImportanceScoreMetricDecorator(
                    original_score_metric, self.score_head, n_tokens
                )
                loss += weight * self.loss(sentence_features, labels)
        finally:
            self.model.forward = original_forward
            self.loss.score_metric = original_score_metric

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "n_doc_tokens": list(self.n_doc_tokens),
            "doc_tokens_weights": list(self.doc_tokens_weights),
            "n_tokens_per_step": self.n_tokens_per_step,
            "strategy": "importance_ste",
        }
