from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from ..models import ColBERT
from ..scores import colbert_scores
from ..utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from .contrastive import extract_skiplist_mask


class RandContext:
    """Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        if torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend is not supported for this operation. Please use CPU or CUDA."
            )
        else:
            self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj,
) -> None:
    """A backward hook that re-runs the forward for each mini-batch with gradients enabled
    and uses the cached partial derivatives w.r.t. the embeddings to backprop.
    """
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(
            sentence_features, loss_obj.cache, loss_obj.random_states
        ):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # Plug back the cached gradients into the backward pass by dotting them with the corresponding representations
                # Scale by grad_output from the top-level backward pass to account for gradient of downstream operations (not useful in this setup)
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


class CachedContrastive(nn.Module):
    """A cached, in-batch negatives contrastive loss for PyLate, analogous to
    SentenceTransformers' CachedMultipleNegativesRankingLoss. This allows
    large effective batch sizes by chunking the embeddings pass and caching
    gradients w.r.t. those embeddings.

    Parameters
    ----------
    model :
        A PyLate ColBERT model
    score_metric
        ColBERT scoring function. Defaults to colbert_scores.
    mini_batch_size
        Chunk size for the forward pass. You can keep this small to avoid OOM on large batch sizes.
    size_average
        Whether to average or sum the cross-entropy loss across the mini-batch.
    gather_across_devices
        Whether to gather the embeddings across devices to have more in batch negatives. We recommand making sure the sampling across GPUs use the same dataset in case of multi-dataset training to make sure the negatives are plausible.
    show_progress_bar
        Whether to show a TQDM progress bar for the embedding steps.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = losses.CachedContrastive(model=model, mini_batch_size=1)

    >>> anchors = model.tokenize([
    ...     "fruits are healthy.", "chips are not healthy."
    ... ], is_query=True)

    >>> positives = model.tokenize([
    ...     "fruits are good for health.", "chips are not good for health."
    ... ], is_query=False)

    >>> negatives = model.tokenize([
    ...     "fruits are bad for health.", "chips are good for health."
    ... ], is_query=False)

    >>> sentence_features = [anchors, positives, negatives]

    >>> loss = loss(sentence_features=sentence_features)
    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_scores,
        mini_batch_size: int = 32,
        size_average: bool = True,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super(CachedContrastive, self).__init__()
        self.model = model
        self.score_metric = score_metric
        self.mini_batch_size = mini_batch_size
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.show_progress_bar = show_progress_bar
        self.temperature = temperature

        # Will hold partial derivatives for each embedding chunk
        self.cache: list[list[Tensor]] | None = None
        # Will hold random states for each chunk, so we can re-run the embedding pass with grads
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Forward pass on a slice [begin:end] of sentence_feature. If 'with_grad' is False,
        we run under torch.no_grad. If 'copy_random_state' is True, we create and return
        a RandContext so that we can exactly reproduce this forward pass later.
        """
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            k: v[begin:end] for k, v in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                # If we need a new random-state copy, create it
                random_state = (
                    RandContext(*sentence_feature_minibatch.values())
                    if copy_random_state
                    else None
                )
                outputs = self.model(sentence_feature_minibatch)
                # by default, PyLate ColBERT forward returns a dict with "token_embeddings"
                embeddings = F.normalize(outputs["token_embeddings"], p=2, dim=-1)

        return embeddings, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Yields chunks of embeddings (and corresponding RandContext) for the given
        sentence_feature, respecting the mini_batch_size limit.
        """
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps, masks) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        # we want partial grads on all the chunked embeddings
        loss = self.calculate_loss(reps, masks, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [
            [r.grad for r in rs] for rs in reps
        ]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps, masks, with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients. Each sub-list in reps is a list of mini-batch chunk embeddings

        Parameters
        ----------
        reps :
            A list of list of mini-batch chunk embeddings. The first list are the anchors, the second are the positives and the remaining are negatives.
        masks
            Tensors containing the skiplist masks assocaited with each sentence feature (anchor, positives, negatives).
        with_backward
            Whether to compute the backward pass or not.
        """
        # We first cat them chunk-wise for anchor, positives, negatives
        embeddings_anchor = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_other = [
            torch.cat([chunk_embed for chunk_embed in r]) for r in reps[1:]
        ]  # [(nneg * bsz, hdim)]

        batch_size = len(embeddings_anchor)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=reps[0][0].device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        # Possibly gather the embeddings across devices to have more in-batch negatives. For GradCache, we only need to gather them to compute the scores matrix and nowhere else.
        # Note that we only gather the documents embeddings and not the queries embeddings, but are keeping gradients. This is to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616
        if self.gather_across_devices:
            embeddings_other = [
                torch.cat(all_gather_with_gradients(embeddings))
                for embeddings in embeddings_other
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
        losses: list[torch.Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            end = begin + self.mini_batch_size
            # We chunk the scores computation to avoid OOM because MaxSim can get expensive with large batch sizes/long documents
            scores = torch.cat(
                [
                    torch.cat(
                        [
                            self.score_metric(
                                embeddings_anchor[begin:end],
                                group_embeddings[
                                    g_start : min(
                                        g_start + self.mini_batch_size,
                                        len(group_embeddings),
                                    )
                                ],
                                mask[
                                    g_start : min(
                                        g_start + self.mini_batch_size,
                                        len(group_embeddings),
                                    )
                                ],
                            )
                            for g_start in range(
                                0, len(group_embeddings), self.mini_batch_size
                            )
                        ],
                        dim=1,
                    )
                    for group_embeddings, mask in zip(embeddings_other, masks[1:])
                ],
                dim=1,
            )
            # We don't want to average the loss across the mini-batch as mini-batch sizes can vary, which would create an issue similar to this one: https://huggingface.co/blog/gradient_accumulation#where-does-it-stem-from
            loss_mbatch = (
                F.cross_entropy(
                    input=scores / self.temperature,
                    target=labels[begin:end],
                    reduction="sum",
                )
                * get_world_size()
            )

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        if self.size_average:
            loss /= batch_size

        return loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the CachedConstrastive loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.
        labels
            The labels for the contrastive loss. Not used in this implementation, but kept for compatibility with Trainer.
        """
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)
        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps, masks)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(
                partial(
                    _backward_hook, sentence_features=sentence_features, loss_obj=self
                )
            )
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, masks)

        return loss

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
