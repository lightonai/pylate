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


class SoftTopKGate(nn.Module):
    """Learned gating module for soft top-k token selection.

    A linear head scores each token, then a sigmoid gate centered on the k-th
    score produces soft weights. During training all tokens remain (weighted);
    at inference hard top-k selection reduces the tensor.

    Parameters
    ----------
    embed_dim
        Embedding dimension of the ColBERT model.
    init_temperature
        Initial value for the learnable temperature scalar.
    """

    def __init__(self, embed_dim: int, init_temperature: float = 5.0) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None,
        k: int,
        hard: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply soft or hard top-k gating.

        Parameters
        ----------
        embeddings
            (batch, T, dim) token embeddings.
        mask
            (batch, T) boolean mask (True = valid token).
        k
            Number of tokens to select.
        hard
            If True, perform hard top-k and return (batch, k, dim).
            If False, soft-weight all T tokens in place.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Gated embeddings and updated mask.
        """
        batch, T, dim = embeddings.shape
        if k >= T:
            return embeddings, mask

        scores = self.linear(embeddings).squeeze(-1)  # (batch, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        if hard:
            # Inference: hard top-k selection
            _, topk_idx = scores.topk(k, dim=1)  # (batch, k)
            topk_idx_sorted, _ = topk_idx.sort(dim=1)
            selected = embeddings.gather(
                1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, dim)
            )
            selected_mask = None
            if mask is not None:
                selected_mask = mask.gather(1, topk_idx_sorted)
            return selected, selected_mask

        # Training: soft gating — all tokens remain, weighted by sigmoid gate.
        # Use topk to find the k-th score as threshold. Detach so gradients
        # flow only through the scores directly (not back through the sort/topk),
        # preventing gradient accumulation at the threshold position.
        topk_scores, _ = scores.topk(k, dim=1)
        threshold = topk_scores[:, -1:].detach()  # (batch, 1), no grad

        diff = scores - threshold
        # When k > number of valid (unmasked) tokens for a batch element,
        # threshold is -inf. Then: valid_score - (-inf) = inf, and
        # (-inf) - (-inf) = NaN. clamp does NOT replace NaN, so we must
        # use nan_to_num first to map inf→50, NaN→-50 (sigmoid→~0).
        diff = diff.nan_to_num(nan=-50.0, posinf=50.0, neginf=-50.0)
        diff = diff.clamp(min=-50.0, max=50.0)
        soft_mask = torch.sigmoid(
            diff * self.temperature
        )  # (batch, T)
        # Zero out masked positions explicitly
        if mask is not None:
            soft_mask = soft_mask * mask.float()
        weighted = embeddings * soft_mask.unsqueeze(-1)
        return weighted, mask


def _apply_soft_topk_3d(
    gate: SoftTopKGate,
    embeddings: Tensor,
    mask: Tensor | None,
    k: int,
    hard: bool,
) -> tuple[Tensor, Tensor | None]:
    """Apply SoftTopKGate to a 3D tensor (batch, T, dim)."""
    return gate(embeddings, mask, k, hard=hard)


def _apply_soft_topk(
    gate: SoftTopKGate,
    embeddings: Tensor,
    mask: Tensor | None,
    k: int,
    hard: bool,
) -> tuple[Tensor, Tensor | None]:
    """Apply SoftTopKGate, handling both 3D and 4D tensors.

    For 4D (batch, n_ways, T, dim), reshapes to 3D, applies, reshapes back.
    """
    if embeddings.dim() == 3:
        return _apply_soft_topk_3d(gate, embeddings, mask, k, hard)
    elif embeddings.dim() == 4:
        batch, n_ways, T, dim = embeddings.shape
        flat_emb = embeddings.reshape(batch * n_ways, T, dim)
        flat_mask = None
        if mask is not None:
            flat_mask = mask.reshape(batch * n_ways, T)
        out_emb, out_mask = _apply_soft_topk_3d(
            gate, flat_emb, flat_mask, k, hard
        )
        out_T = out_emb.shape[1]
        out_emb = out_emb.reshape(batch, n_ways, out_T, dim)
        if out_mask is not None:
            out_mask = out_mask.reshape(batch, n_ways, out_T)
        return out_emb, out_mask
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {embeddings.dim()}D")


class SoftTopKScoreMetricDecorator:
    """Wraps a ColBERT score function to apply soft top-k gating on document
    embeddings before scoring."""

    def __init__(
        self,
        score_fn,
        gate: SoftTopKGate,
        n_tokens: int,
        hard: bool = False,
    ) -> None:
        self.score_fn = score_fn
        self.gate = gate
        self.n_tokens = n_tokens
        self.hard = hard

    def __call__(
        self,
        queries_embeddings,
        documents_embeddings,
        queries_mask=None,
        documents_mask=None,
    ):
        documents_embeddings, documents_mask = _apply_soft_topk(
            self.gate,
            documents_embeddings,
            documents_mask,
            self.n_tokens,
            hard=self.hard,
        )
        return self.score_fn(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )


class CachedSoftTopKLossDecorator:
    """Decorator for CachedContrastive.calculate_loss that applies soft top-k
    gating at multiple document token counts.

    Follows the same gradient-chaining pattern as CachedDocTokensLossDecorator.
    """

    def __init__(
        self,
        fn,
        gate: SoftTopKGate,
        n_doc_tokens: Sequence[int],
        doc_tokens_weights: Sequence[float] | Sequence[int],
        n_tokens_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.gate = gate
        self.n_doc_tokens = n_doc_tokens
        self.doc_tokens_weights = doc_tokens_weights
        self.n_tokens_per_step = n_tokens_per_step

    def __call__(
        self, reps: list[list[Tensor]], masks: list[Tensor], *args, **kwargs
    ) -> Tensor:
        dim_indices = list(range(len(self.n_doc_tokens)))
        if 0 < self.n_tokens_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_tokens_per_step)

        hard = not torch.is_grad_enabled()
        loss = 0.0
        for idx in dim_indices:
            n_tokens = self.n_doc_tokens[idx]
            weight = self.doc_tokens_weights[idx]

            # Apply soft top-k gating to document reps and masks
            gated = []
            gated_masks = [masks[0]]  # query mask unchanged
            for i, sent_reps in enumerate(reps):
                if i == 0:
                    # Clone query reps to create non-leaf tensors for gradient chaining
                    gated.append([r.clone() for r in sent_reps])
                else:
                    gated_sent = []
                    for r in sent_reps:
                        # sent_reps are 3D (minibatch, T, dim) chunks
                        gated_r, _ = _apply_soft_topk_3d(
                            self.gate, r, None, n_tokens, hard=hard
                        )
                        gated_sent.append(gated_r)
                    gated.append(gated_sent)

            # For document masks, apply gating with mask
            for doc_mask in masks[1:]:
                if hard:
                    # Need to figure out which tokens were selected
                    # Since we don't have per-chunk mask alignment, use truncation as fallback
                    # For hard mode, soft_topk on embeddings already selected tokens,
                    # but masks need matching. We apply the gate to get indices.
                    dummy_emb = torch.zeros(
                        doc_mask.shape[0],
                        doc_mask.shape[1],
                        1,
                        device=doc_mask.device,
                    )
                    _, gated_m = _apply_soft_topk_3d(
                        self.gate, dummy_emb, doc_mask, n_tokens, hard=True
                    )
                    gated_masks.append(
                        gated_m if gated_m is not None else doc_mask
                    )
                else:
                    # Soft mode: all tokens remain, mask unchanged
                    gated_masks.append(doc_mask)

            compute_gradients = torch.is_grad_enabled()
            if compute_gradients:
                matryoshka_reps = [
                    [r.detach().requires_grad_() for r in sent_reps]
                    for sent_reps in gated
                ]
            else:
                matryoshka_reps = gated

            loss += weight * self.fn(
                matryoshka_reps, gated_masks, *args, **kwargs
            )

            if compute_gradients:
                for t_minibatch, d_minibatch in zip(gated, matryoshka_reps):
                    for t, d in zip(t_minibatch, d_minibatch):
                        if d.grad is not None:
                            t.backward(weight * d.grad)

        return loss


class MatryoshkaSoftTopKLoss(nn.Module):
    """Matryoshka-style loss with learned soft top-k gating for document token
    selection.

    During training, a differentiable sigmoid gate soft-weights all document
    tokens (no STE needed). Tokens near the top-k boundary receive weights
    close to 1, tokens far below receive weights close to 0. At inference,
    hard top-k selection physically reduces the tensor.

    The gate module (``nn.Linear(embed_dim, 1)`` + learnable temperature)
    lives on this loss and is discovered by ``SentenceTransformerTrainer``
    via ``named_parameters()``.

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
    init_temperature
        Initial temperature for the sigmoid gating.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> base_loss = losses.Distillation(model=model)

    >>> loss = losses.MatryoshkaSoftTopKLoss(
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
        init_temperature: float = 5.0,
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
        self.gate = SoftTopKGate(
            embed_dim=embed_dim, init_temperature=init_temperature
        )
        # Move gate to model device
        device = next(model.parameters()).device
        self.gate = self.gate.to(device)

        if isinstance(loss, CachedContrastive):
            loss.calculate_loss = CachedSoftTopKLossDecorator(
                loss.calculate_loss,
                self.gate,
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

            hard = not self.training
            dim_indices = list(range(len(self.n_doc_tokens)))
            if 0 < self.n_tokens_per_step < len(dim_indices):
                dim_indices = sorted(
                    random.sample(dim_indices, self.n_tokens_per_step)
                )

            loss = 0.0
            for idx in dim_indices:
                n_tokens = self.n_doc_tokens[idx]
                weight = self.doc_tokens_weights[idx]
                self.loss.score_metric = SoftTopKScoreMetricDecorator(
                    original_score_metric, self.gate, n_tokens, hard=hard
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
            "strategy": "soft_topk",
        }
