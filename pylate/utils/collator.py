from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from sentence_transformers.base.data_collator import BaseDataCollator

logger = logging.getLogger(__name__)


@dataclass
class ColBERTCollator(BaseDataCollator):
    """Collator for ColBERT model.

    Extends :class:`BaseDataCollator` with ColBERT-specific behavior:
    - Detects query vs document columns (``is_query``) based on column name
    - Flattens list-of-list columns (e.g. multiple documents per query)
    - Skips ``_id`` columns
    - Always passes ``pad=True`` to the preprocess function
    - Optionally tracks prompt lengths for prompt-aware pooling

    Parameters
    ----------
    preprocess_fn
        The function to preprocess/tokenize the input text. This should be
        ``model.preprocess`` (or the deprecated ``model.tokenize``).
    tokenize_fn
        Deprecated alias for ``preprocess_fn``. Use ``preprocess_fn`` instead.
    valid_label_columns
        The name of the columns that contain the labels: scores or labels.
    router_mapping
        The mapping of the columns to the router.
    prompts
        The prompts to use for the columns.
    include_prompt_lengths
        Whether to include the prompt lengths in the batch.
    all_special_ids
        The special ids to use for the tokenization.

    Examples
    --------
    >>> from pylate import models, utils

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> collator = utils.ColBERTCollator(
    ...     preprocess_fn=model.preprocess,
    ... )

    >>> features = [
    ...     {
    ...         "query": "fruits are healthy.",
    ...         "positive": "fruits are good for health.",
    ...         "negative": "fruits are bad for health.",
    ...         "label": [0.7, 0.3]
    ...     }
    ... ]

    >>> features = collator(features=features)

    >>> fields = [
    ...     "query_input_ids",
    ...     "positive_input_ids",
    ...     "negative_input_ids",
    ...     "query_attention_mask",
    ...     "positive_attention_mask",
    ...     "negative_attention_mask",
    ...     "query_token_type_ids",
    ...     "positive_token_type_ids",
    ...     "negative_token_type_ids",
    ... ]

    >>> for field in fields:
    ...     assert field in features
    ...     assert isinstance(features[field], torch.Tensor)
    ...     assert features[field].ndim == 2

    """

    include_prompt_lengths: bool = False
    all_special_ids: set[int] = field(default_factory=set)
    _prompt_length_mapping: dict[tuple[str, str | None], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __init__(
        self,
        preprocess_fn: Callable | None = None,
        tokenize_fn: Callable | None = None,
        valid_label_columns: list[str] | None = None,
        router_mapping: dict[str, str] | dict[str, dict[str, str]] | None = None,
        prompts: dict[str, str] | dict[str, dict[str, str]] | None = None,
        include_prompt_lengths: bool = False,
        all_special_ids: set[int] | None = None,
        _prompt_length_mapping: dict | None = None,
        _warned_columns: set[tuple[str]] | None = None,
    ):
        # Support both preprocess_fn and deprecated tokenize_fn
        if preprocess_fn is not None:
            actual_fn = preprocess_fn
        elif tokenize_fn is not None:
            actual_fn = tokenize_fn
        else:
            raise ValueError("Either `preprocess_fn` or `tokenize_fn` must be provided.")

        if valid_label_columns is None:
            valid_label_columns = ["label", "scores"]

        # Initialize the dataclass fields via the parent
        super().__init__(
            preprocess_fn=actual_fn,
            valid_label_columns=valid_label_columns,
            router_mapping=router_mapping if router_mapping is not None else {},
            prompts=prompts if prompts is not None else {},
        )

        # Keep backward compat alias
        self.tokenize_fn = self.preprocess_fn

        self.include_prompt_lengths = include_prompt_lengths
        self.all_special_ids = all_special_ids if all_special_ids is not None else set()
        self._prompt_length_mapping = (
            _prompt_length_mapping if _prompt_length_mapping is not None else {}
        )
        if _warned_columns is not None:
            self._warned_columns = _warned_columns

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """Collate a list of features into a batch."""
        if not features:
            return {}

        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {"return_loss": True}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists (inherited logic)
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        router_mapping = self._resolve_router_mapping(batch)
        prompts = self._resolve_prompts(batch)

        for column_name in column_names:
            # We do not tokenize columns containing the ids.
            if "_id" in column_name:
                continue

            task = router_mapping.get(column_name, None)
            texts = [row[column_name] for row in features]
            # Flatten the list of texts if it is a list of lists (e.g, documents)
            if isinstance(texts[0], list):
                texts = list(itertools.chain(*texts))

            # Detect if inputs are multimodal (non-text)
            is_multimodal = not self._is_text_column(texts)

            # Get the string prompt for the column, if it exists.
            prompt = self._get_prompt_for_column(prompts, column_name)

            # If a prompt is provided, we prepend it to the column values.
            if prompt and not is_multimodal:
                if self.include_prompt_lengths:
                    prompt_length = self._get_prompt_length(prompt, task=task)
                    if prompt_length is not None:
                        batch[f"{column_name}_prompt_length"] = torch.tensor(
                            [prompt_length] * len(features), dtype=torch.int
                        )
                texts = [prompt + text for text in texts]

            is_query = "query" in column_name or "anchor" in column_name
            tokenized = self.preprocess_fn(texts, is_query=is_query, pad=True, task=task)

            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch

    @staticmethod
    def _is_text_column(values: list) -> bool:
        """Check if column values are text strings."""
        if not values:
            return True
        first = values[0]
        return isinstance(first, str)

    def _get_prompt_length(self, prompt: str, task: str | None = None) -> int:
        if (prompt, task) in self._prompt_length_mapping:
            return self._prompt_length_mapping[(prompt, task)]

        tokenized_prompt = self.preprocess_fn([prompt], task=task)
        if "input_ids" not in tokenized_prompt:
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a special EOS token, we do not count it as part of the prompt length.
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if last_token in self.all_special_ids:
            prompt_length -= 1

        self._prompt_length_mapping[(prompt, task)] = prompt_length
        return prompt_length
