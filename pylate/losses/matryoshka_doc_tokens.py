from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..models import ColBERT
from .cached_contrastive import CachedContrastive

logger = logging.getLogger(__name__)


def truncate_doc_tokens(tensor: Tensor, n_tokens: int) -> Tensor:
    """Truncate a tensor along the token dimension.

    For 3D tensors (batch_size, tokens, dim): returns (batch_size, n_tokens, dim).
    For 4D tensors (batch_size, n_ways, tokens, dim): returns (batch_size, n_ways, n_tokens, dim).
    If n_tokens exceeds the actual token count, the tensor is returned unchanged.
    """
    if tensor.dim() == 3:
        return tensor[:, :n_tokens, :]
    elif tensor.dim() == 4:
        return tensor[:, :, :n_tokens, :]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")


def truncate_mask(mask: Tensor, n_tokens: int) -> Tensor:
    """Truncate a mask tensor along the token dimension.

    For 2D masks (batch_size, tokens): returns (batch_size, n_tokens).
    For 3D masks (batch_size, n_ways, tokens): returns (batch_size, n_ways, n_tokens).
    """
    if mask.dim() == 2:
        return mask[:, :n_tokens]
    elif mask.dim() == 3:
        return mask[:, :, :n_tokens]
    else:
        raise ValueError(f"Expected 2D or 3D mask, got {mask.dim()}D")


class ForwardCachingDecorator:
    """Caches model.forward outputs to avoid recomputation across token count iterations.

    Unlike MatryoshkaLoss's ForwardDecorator, this does not modify the output -
    it purely caches. Document token truncation happens at the score metric level.
    """

    def __init__(self, fn) -> None:
        self.fn = fn
        self.cache = {}

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        key = id(features["input_ids"])
        if key not in self.cache:
            self.cache[key] = self.fn(features)
        return self.cache[key]


class ScoreMetricDecorator:
    """Wraps a ColBERT score function to truncate document token embeddings
    and masks before computing MaxSim scores."""

    def __init__(self, score_fn, n_tokens: int) -> None:
        self.score_fn = score_fn
        self.n_tokens = n_tokens

    def __call__(
        self,
        queries_embeddings,
        documents_embeddings,
        queries_mask=None,
        documents_mask=None,
    ):
        documents_embeddings = truncate_doc_tokens(
            documents_embeddings, self.n_tokens
        )
        if documents_mask is not None:
            documents_mask = truncate_mask(documents_mask, self.n_tokens)
        return self.score_fn(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )


class CachedDocTokensLossDecorator:
    """Decorator for CachedContrastive.calculate_loss that computes the loss
    at multiple document token counts.

    Follows the same pattern as sentence_transformers' CachedLossDecorator for
    MatryoshkaLoss, but truncates along the token dimension instead of the
    embedding dimension.
    """

    def __init__(
        self,
        fn,
        n_doc_tokens: Sequence[int],
        doc_tokens_weights: Sequence[float] | Sequence[int],
        n_tokens_per_step: int = -1,
    ) -> None:
        self.fn = fn
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

            # Truncate reps: query reps (index 0) are cloned to create non-leaf
            # tensors for gradient chaining; doc reps (index 1+) are truncated.
            truncated = []
            for i, sent_reps in enumerate(reps):
                if i == 0:
                    truncated.append([r.clone() for r in sent_reps])
                else:
                    truncated.append(
                        [truncate_doc_tokens(r, n_tokens) for r in sent_reps]
                    )

            # Truncate document masks (keep query mask unchanged)
            truncated_masks = [masks[0]]
            for doc_mask in masks[1:]:
                truncated_masks.append(truncate_mask(doc_mask, n_tokens))

            compute_gradients = torch.is_grad_enabled()
            if compute_gradients:
                # Detach truncated reps so each iteration gets its own backward;
                # gradients are chained back to the originals below.
                matryoshka_reps = [
                    [r.detach().requires_grad_() for r in sent_reps]
                    for sent_reps in truncated
                ]
            else:
                matryoshka_reps = truncated

            loss += weight * self.fn(
                matryoshka_reps, truncated_masks, *args, **kwargs
            )

            # Chain gradients back through the clone/truncation operations
            if compute_gradients:
                for t_minibatch, d_minibatch in zip(truncated, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        if d.grad is not None:
                            t.backward(weight * d.grad)

        return loss


class MatryoshkaDocTokensLoss(nn.Module):
    """Matryoshka-style loss that trains a multi-vector model (e.g. ColBERT) to
    perform well when using only a subset of document token embeddings for MaxSim
    scoring.

    This enables a storage-performance tradeoff at retrieval time: store fewer
    document token embeddings per document to reduce index size, while maintaining
    reasonable retrieval quality.

    The loss wraps any PyLate loss (Contrastive, Distillation, CachedContrastive)
    and computes it at multiple document token counts, summing the weighted losses.

    Parameters
    ----------
    model
        ColBERT model.
    loss
        Base loss function (e.g. Distillation, Contrastive, CachedContrastive).
    n_doc_tokens
        List of document token counts to train at, e.g. [32, 64, 128, 256].
        It is recommended to include the full document length to ensure no
        degradation when using all tokens at retrieval time.
    doc_tokens_weights
        Weights for each token count's loss contribution. Defaults to all 1.0.
    n_tokens_per_step
        Number of token counts to randomly sample per training step.
        -1 means use all token counts every step.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> base_loss = losses.Distillation(model=model)

    >>> loss = losses.MatryoshkaDocTokensLoss(
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

        # Sort descending (largest first, like MatryoshkaLoss)
        dims_weights = sorted(
            zip(n_doc_tokens, doc_tokens_weights),
            key=lambda x: x[0],
            reverse=True,
        )
        self.n_doc_tokens: tuple[int, ...]
        self.doc_tokens_weights: tuple[float, ...] | tuple[int, ...]
        self.n_doc_tokens, self.doc_tokens_weights = zip(*dims_weights)
        self.n_tokens_per_step = n_tokens_per_step

        # Handle CachedContrastive with a decorator on calculate_loss,
        # following the same pattern as MatryoshkaLoss's CachedLossDecorator.
        if isinstance(loss, CachedContrastive):
            loss.calculate_loss = CachedDocTokensLossDecorator(
                loss.calculate_loss,
                self.n_doc_tokens,
                self.doc_tokens_weights,
                self.n_tokens_per_step,
            )

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        # For CachedContrastive, the decorator on calculate_loss handles everything.
        if isinstance(self.loss, CachedContrastive):
            return self.loss(sentence_features, labels)

        # For non-cached losses: cache model.forward outputs and swap score_metric
        # with a truncating wrapper for each document token count.
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
                self.loss.score_metric = ScoreMetricDecorator(
                    original_score_metric, n_tokens
                )
                loss += weight * self.loss(sentence_features, labels)
        finally:
            self.model.forward = original_forward
            self.loss.score_metric = original_score_metric

        return loss

    def get_doc_token_reducer(self):
        """Return a callable that reduces document tokens via positional truncation.

        Returns a function ``(embeddings, n_tokens) -> embeddings[:, :n_tokens, :]``
        suitable for passing as ``doc_token_reducer`` to ``NanoBEIREvaluator``.
        """

        def reducer(embeddings: Tensor, n_tokens: int) -> Tensor:
            return embeddings[:, :n_tokens, :]

        return reducer

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "n_doc_tokens": list(self.n_doc_tokens),
            "doc_tokens_weights": list(self.doc_tokens_weights),
            "n_tokens_per_step": self.n_tokens_per_step,
        }
