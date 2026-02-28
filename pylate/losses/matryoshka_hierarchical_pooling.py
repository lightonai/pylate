from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..models import ColBERT
from .cached_contrastive import CachedContrastive
from .matryoshka_doc_tokens import ForwardCachingDecorator, truncate_doc_tokens, truncate_mask

logger = logging.getLogger(__name__)


class LearnedPoolingLevel(nn.Module):
    """One level of learned hierarchical pooling.

    Groups consecutive tokens into pairs (stride=2) and uses a learned
    attention gate to produce weighted averages, halving the token count.

    Parameters
    ----------
    embed_dim
        Embedding dimension of the ColBERT model.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Pool T tokens down to ~T/2 via learned attention over pairs.

        Parameters
        ----------
        embeddings
            (batch, T, dim) token embeddings.
        mask
            (batch, T) boolean mask (True = valid).

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Pooled embeddings (batch, ceil(T/2), dim) and mask.
        """
        batch, T, dim = embeddings.shape

        # Number of complete pairs
        K = T // 2
        remainder = T % 2

        if K == 0:
            # Only one token, can't pool further
            return embeddings, mask

        # Group consecutive tokens into pairs
        paired = embeddings[:, : K * 2].reshape(batch, K, 2, dim)

        # Compute gate logits
        gate_logits = self.gate(paired).squeeze(-1)  # (batch, K, 2)

        # Mask padding within groups
        if mask is not None:
            pair_mask = mask[:, : K * 2].reshape(batch, K, 2)
            gate_logits = gate_logits.masked_fill(~pair_mask, float("-inf"))
            # Pooled mask: a group is valid if any token in it was valid
            pooled_mask = pair_mask.any(dim=2)  # (batch, K)
        else:
            pair_mask = None
            pooled_mask = None

        # Learned attention within each pair
        weights = torch.softmax(gate_logits, dim=2)  # (batch, K, 2)

        # Handle case where both tokens in a pair are masked (softmax of [-inf, -inf] = nan)
        if pair_mask is not None:
            all_masked = ~pair_mask.any(dim=2, keepdim=True)  # (batch, K, 1)
            weights = weights.masked_fill(all_masked, 0.0)

        pooled = (paired * weights.unsqueeze(-1)).sum(dim=2)  # (batch, K, dim)

        # Append remainder token if T is odd
        if remainder > 0:
            remainder_emb = embeddings[:, -1:, :]  # (batch, 1, dim)
            pooled = torch.cat([pooled, remainder_emb], dim=1)
            if mask is not None:
                remainder_mask = mask[:, -1:]  # (batch, 1)
                pooled_mask = torch.cat([pooled_mask, remainder_mask], dim=1)

        return pooled, pooled_mask


class HierarchicalPoolingStack(nn.Module):
    """Stack of learned pooling levels that iteratively halve the token count.

    For n_doc_tokens=[32, 64, 128, 256] with stride 2, we need
    ceil(log2(256/32)) = 3 levels to go from 256 → 128 → 64 → 32.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    n_levels
        Number of pooling levels (each halves the token count).
    """

    def __init__(self, embed_dim: int, n_levels: int) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [LearnedPoolingLevel(embed_dim) for _ in range(n_levels)]
        )

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None,
        target_n_tokens: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Pool embeddings down to target_n_tokens.

        Applies pooling levels one at a time until the token count is at or
        below the target. If the result overshoots, a final positional
        truncation is applied.

        Parameters
        ----------
        embeddings
            (batch, T, dim) token embeddings.
        mask
            (batch, T) boolean mask.
        target_n_tokens
            Desired number of output tokens.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Pooled embeddings and mask.
        """
        current_T = embeddings.shape[1]

        if current_T <= target_n_tokens:
            return embeddings, mask

        for level in self.levels:
            if current_T <= target_n_tokens:
                break
            embeddings, mask = level(embeddings, mask)
            current_T = embeddings.shape[1]

        # Final truncation if we overshot
        if current_T > target_n_tokens:
            embeddings = embeddings[:, :target_n_tokens, :]
            if mask is not None:
                mask = mask[:, :target_n_tokens]

        return embeddings, mask


def _compute_n_levels(n_doc_tokens: Sequence[int]) -> int:
    """Compute the number of pooling levels needed.

    We need enough levels to halve from the largest token count down to the
    smallest: ceil(log2(max / min)).
    """
    max_tokens = max(n_doc_tokens)
    min_tokens = min(n_doc_tokens)
    if min_tokens <= 0:
        raise ValueError("All n_doc_tokens must be > 0")
    if max_tokens == min_tokens:
        return 1
    return max(1, math.ceil(math.log2(max_tokens / min_tokens)))


def _pool_mask(mask: Tensor, target_n_tokens: int, n_levels: int) -> Tensor:
    """Pool a mask by structurally halving via pairing with `any`.

    This mirrors the structural transformation of HierarchicalPoolingStack
    without needing the gate's linear layer (which requires full embeddings).

    Parameters
    ----------
    mask
        (batch, T) boolean mask.
    target_n_tokens
        Desired number of output tokens.
    n_levels
        Number of pooling levels available.

    Returns
    -------
    Tensor
        Pooled mask (batch, target_T).
    """
    current_T = mask.shape[1]
    if current_T <= target_n_tokens:
        return mask

    for _ in range(n_levels):
        if current_T <= target_n_tokens:
            break
        K = current_T // 2
        remainder = current_T % 2
        if K == 0:
            break
        pair_mask = mask[:, : K * 2].reshape(mask.shape[0], K, 2)
        pooled_mask = pair_mask.any(dim=2)
        if remainder > 0:
            pooled_mask = torch.cat(
                [pooled_mask, mask[:, -1:]], dim=1
            )
        mask = pooled_mask
        current_T = mask.shape[1]

    if current_T > target_n_tokens:
        mask = mask[:, :target_n_tokens]

    return mask


def _apply_pooling_3d(
    pool_stack: HierarchicalPoolingStack,
    embeddings: Tensor,
    mask: Tensor | None,
    target: int,
) -> tuple[Tensor, Tensor | None]:
    """Apply HierarchicalPoolingStack to a 3D tensor."""
    return pool_stack(embeddings, mask, target)


def _apply_pooling(
    pool_stack: HierarchicalPoolingStack,
    embeddings: Tensor,
    mask: Tensor | None,
    target: int,
) -> tuple[Tensor, Tensor | None]:
    """Apply pooling, handling both 3D and 4D tensors."""
    if embeddings.dim() == 3:
        return _apply_pooling_3d(pool_stack, embeddings, mask, target)
    elif embeddings.dim() == 4:
        batch, n_ways, T, dim = embeddings.shape
        flat_emb = embeddings.reshape(batch * n_ways, T, dim)
        flat_mask = None
        if mask is not None:
            flat_mask = mask.reshape(batch * n_ways, T)
        out_emb, out_mask = _apply_pooling_3d(
            pool_stack, flat_emb, flat_mask, target
        )
        out_T = out_emb.shape[1]
        out_emb = out_emb.reshape(batch, n_ways, out_T, dim)
        if out_mask is not None:
            out_mask = out_mask.reshape(batch, n_ways, out_T)
        return out_emb, out_mask
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {embeddings.dim()}D")


class HierarchicalPoolingScoreMetricDecorator:
    """Wraps a ColBERT score function to apply hierarchical pooling on document
    embeddings before scoring."""

    def __init__(
        self,
        score_fn,
        pool_stack: HierarchicalPoolingStack,
        n_tokens: int,
    ) -> None:
        self.score_fn = score_fn
        self.pool_stack = pool_stack
        self.n_tokens = n_tokens

    def __call__(
        self,
        queries_embeddings,
        documents_embeddings,
        queries_mask=None,
        documents_mask=None,
    ):
        documents_embeddings, documents_mask = _apply_pooling(
            self.pool_stack,
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


class CachedHierarchicalPoolingLossDecorator:
    """Decorator for CachedContrastive.calculate_loss that applies
    hierarchical pooling at multiple document token counts.

    Follows the same gradient-chaining pattern as CachedDocTokensLossDecorator.
    """

    def __init__(
        self,
        fn,
        pool_stack: HierarchicalPoolingStack,
        n_doc_tokens: Sequence[int],
        doc_tokens_weights: Sequence[float] | Sequence[int],
        n_tokens_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.pool_stack = pool_stack
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

            # Apply hierarchical pooling to document reps and masks
            pooled = []
            pooled_masks = [masks[0]]  # query mask unchanged
            for i, sent_reps in enumerate(reps):
                if i == 0:
                    pooled.append([r.clone() for r in sent_reps])
                else:
                    pooled_sent = []
                    for r in sent_reps:
                        pooled_r, _ = _apply_pooling_3d(
                            self.pool_stack, r, None, n_tokens
                        )
                        pooled_sent.append(pooled_r)
                    pooled.append(pooled_sent)

            # Pool document masks structurally (halving via pairing)
            n_levels = len(self.pool_stack.levels)
            for doc_mask in masks[1:]:
                pooled_masks.append(
                    _pool_mask(doc_mask, n_tokens, n_levels)
                )

            compute_gradients = torch.is_grad_enabled()
            if compute_gradients:
                matryoshka_reps = [
                    [r.detach().requires_grad_() for r in sent_reps]
                    for sent_reps in pooled
                ]
            else:
                matryoshka_reps = pooled

            loss += weight * self.fn(
                matryoshka_reps, pooled_masks, *args, **kwargs
            )

            if compute_gradients:
                for t_minibatch, d_minibatch in zip(pooled, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        if d.grad is not None:
                            t.backward(weight * d.grad)

        return loss


class MatryoshkaHierarchicalPoolingLoss(nn.Module):
    """Matryoshka-style loss with learned hierarchical pooling for document
    token reduction.

    Iteratively halves the document token count via learned attention-weighted
    pooling of consecutive token pairs: N -> N/2 -> N/4 -> N/8. Each level
    uses a ``Linear(embed_dim, 1)`` gate to learn which token in each pair
    is more important.

    Key advantages:
    - Actual token reduction during training (like Importance+STE)
    - Fully differentiable (like Soft Top-K, no STE needed)
    - Preserves information from ALL original tokens via pooling

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

    >>> loss = losses.MatryoshkaHierarchicalPoolingLoss(
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

        # Compute number of pooling levels needed
        n_levels = _compute_n_levels(self.n_doc_tokens)
        embed_dim = model.get_sentence_embedding_dimension()
        self.pool_stack = HierarchicalPoolingStack(
            embed_dim=embed_dim, n_levels=n_levels
        )
        # Move to model device
        device = next(model.parameters()).device
        self.pool_stack = self.pool_stack.to(device)

        if isinstance(loss, CachedContrastive):
            loss.calculate_loss = CachedHierarchicalPoolingLossDecorator(
                loss.calculate_loss,
                self.pool_stack,
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
                self.loss.score_metric = HierarchicalPoolingScoreMetricDecorator(
                    original_score_metric, self.pool_stack, n_tokens
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
            "strategy": "hierarchical_pooling",
            "n_levels": len(self.pool_stack.levels),
        }
