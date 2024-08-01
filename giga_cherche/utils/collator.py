from typing import Callable

import torch


class ColBERTCollator:
    """Collator for ColBERT model.

    Parameters
    ----------
    tokenize_fn
        The function to tokenize the input text.
    valid_label_columns
        The name of the columns that contain the labels: scores or labels.

    Examples
    --------
    >>> from giga_cherche import models, utils

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> collator = utils.ColBERTCollator(
    ...     tokenize_fn=model.tokenize,
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

    def __init__(
        self, tokenize_fn: Callable, valid_label_columns: list[str] | None = None
    ) -> None:
        self.tokenize_fn = tokenize_fn

        if valid_label_columns is None:
            valid_label_columns = ["label", "score"]

        self.valid_label_columns = valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of features into a batch."""
        batch = {"return_loss": True}

        columns = set(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the target label.
        for label_column in self.valid_label_columns:
            if label_column in columns:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                columns.remove(label_column)
                break

        # Tokenize the text.
        for column in columns:
            if "_id" in column:
                # Skip the columns that contains _id.
                continue

            is_query = "query" in column or "anchor" in column

            tokenized = self.tokenize_fn(
                [row[column] for row in features],
                is_query=is_query,
                pad_document=True,
            )

            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value

        return batch
