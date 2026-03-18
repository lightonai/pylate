from __future__ import annotations

import ast
import logging
import random

import datasets

logger = logging.getLogger(__name__)


class KDProcessing:
    """Dataset processing class for knowledge distillation training.

    Parameters
    ----------
    queries
        Queries dataset.
    documents
        Documents dataset.
    split
        Split to use for the queries and documents datasets. Used only if the queries and documents are of type `datasets.DatasetDict`.
    n_ways
        Number of scores to keep for the distillation.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from pylate import utils

    >>> train = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="train",
    ...    split="train",
    ... )

    >>> queries = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="queries",
    ...    split="train",
    ... )

    >>> documents = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="documents",
    ...    split="train",
    ... )

    >>> train.set_transform(
    ...    utils.KDProcessing(
    ...        queries=queries, documents=documents
    ...    ).transform,
    ... )

    >>> for sample in train:
    ...     assert "documents" in sample and isinstance(sample["documents"], list)
    ...     assert "query" in sample and isinstance(sample["query"], str)
    ...     assert "scores" in sample and isinstance(sample["scores"], list)

    """

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        n_ways: int = 32,
    ) -> None:
        if isinstance(queries, datasets.DatasetDict):
            self.queries = queries[split]
        else:
            self.queries = queries

        if isinstance(documents, datasets.DatasetDict):
            self.documents = documents[split]
        else:
            self.documents = documents

        self.n_ways = n_ways

        self.queries_index = {
            query_id: i for i, query_id in enumerate(iterable=self.queries["query_id"])
        }

        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(iterable=self.documents["document_id"])
        }

    def transform(self, examples: dict) -> dict:
        """Update the input dataset with the queries and documents."""
        if isinstance(examples["scores"][0], str):
            examples["scores"] = [
                ast.literal_eval(node_or_string=score) for score in examples["scores"]
            ]

        examples["scores"] = [score[: self.n_ways] for score in examples["scores"]]

        if isinstance(examples["document_ids"][0], str):
            examples["document_ids"] = [
                ast.literal_eval(node_or_string=document_ids)
                for document_ids in examples["document_ids"]
            ]

        examples["document_ids"] = [
            document_ids[: self.n_ways] for document_ids in examples["document_ids"]
        ]

        examples["query"] = [
            self.queries[self.queries_index[query_id]]["text"]
            for query_id in examples["query_id"]
        ]

        examples["documents"] = []
        for document_ids in examples["document_ids"]:
            documents = []
            for document_id in document_ids:
                try:
                    documents.append(
                        self.documents[self.documents_index[document_id]]["text"]
                    )
                except KeyError:
                    documents.append("")
                    logger.warning(f"Unable to find document: {document_id}")

            examples["documents"].append(documents)

        return examples

    def map(self, example: dict) -> dict:
        """Process a single example.

        Parameters
        ----------
        example
            Example to process.

        Examples
        --------
        >>> from datasets import load_dataset
        >>> from pylate import utils

        >>> train = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="train",
        ...    split="train",
        ... )

        >>> queries = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="queries",
        ...    split="train",
        ... )

        >>> documents = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="documents",
        ...    split="train",
        ... )

        >>> train = train.map(
        ...    utils.KDProcessing(
        ...        queries=queries, documents=documents
        ...    ).map,
        ... )

        >>> for sample in train:
        ...     assert "documents" in sample and isinstance(sample["documents"], list)
        ...     assert "query" in sample and isinstance(sample["query"], str)
        ...     assert "scores" in sample and isinstance(sample["scores"], list)


        """
        if isinstance(example["scores"], str):
            example["scores"] = ast.literal_eval(node_or_string=example["scores"])

        example["scores"] = example["scores"][: self.n_ways]

        if isinstance(example["document_ids"], str):
            example["document_ids"] = ast.literal_eval(
                node_or_string=example["document_ids"]
            )

        example["document_ids"] = example["document_ids"][: self.n_ways]

        processed_example = {
            "scores": example["scores"],
            "query": self.queries[self.queries_index[example["query_id"]]]["text"],
        }

        documents = []
        for document_id in example["document_ids"]:
            try:
                documents.append(
                    self.documents[self.documents_index[document_id]]["text"]
                )
            except KeyError:
                documents.append("")
                logger.warning(f"Unable to find document: {document_id}")

        processed_example["documents"] = documents

        return processed_example


def _passage_text(passage: dict) -> str:
    """Extract text from a passage dict, prepending title if present."""
    title = passage.get("title", "")
    text = passage["text"]
    if title and title.strip():
        return f"{title} {text}"
    return text


class RLHNProcessing:
    """Dataset processing class for the RLHN-680K cleaned negatives dataset.

    Converts the format (query, positive_passages, negative_passages)
    to the flat (query, positive, negative) format expected by
    :class:`~pylate.utils.ColBERTCollator` for contrastive training.

    Positives and negatives are randomly sampled on each access, so different
    combinations are seen across epochs.

    Parameters
    ----------
    n_ways
        Number of negatives per query. When 1 (default), ``negative`` is a
        plain string (standard triplet). When > 1, ``negative`` is a list
        of strings which :class:`~pylate.utils.ColBERTCollator` flattens
        automatically.
    seed
        Random seed for reproducible negative/positive sampling.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from pylate import utils

    >>> train = load_dataset(
    ...     "rlhn/rlhn-680K",
    ...     split="train",
    ... )

    To filter to specific subsets, use ``.filter()`` before ``set_transform``:

    >>> train = train.filter(lambda x: x["subset"] in {"nq", "fever"})

    >>> train.set_transform(
    ...     utils.RLHNProcessing(n_ways=1).transform,
    ... )

    >>> sample = train[0]
    >>> assert "query" in sample and isinstance(sample["query"], str)
    >>> assert "positive" in sample and isinstance(sample["positive"], str)
    >>> assert "negative" in sample

    """

    def __init__(
        self,
        n_ways: int = 1,
        seed: int = 42,
    ) -> None:
        if n_ways < 1:
            raise ValueError(f"n_ways must be >= 1, got {n_ways}")

        self.n_ways = n_ways
        self.rng = random.Random(seed)

    def transform(self, examples: dict) -> dict:
        """Batch transform for use with ``dataset.set_transform``."""
        queries = []
        positives = []
        negatives = []

        for i in range(len(examples["query"])):
            pos_passages = examples["positive_passages"][i]
            neg_passages = examples["negative_passages"][i]

            # Sample one positive
            pos = pos_passages[self.rng.randint(0, len(pos_passages) - 1)]
            positives.append(_passage_text(pos))

            # Sample n_ways negatives
            n_avail = len(neg_passages)
            if n_avail >= self.n_ways:
                sampled = self.rng.sample(neg_passages, self.n_ways)
            else:
                sampled = self.rng.choices(neg_passages, k=self.n_ways)

            neg_texts = [_passage_text(neg) for neg in sampled]
            negatives.append(neg_texts[0] if self.n_ways == 1 else neg_texts)

            queries.append(examples["query"][i])

        return {"query": queries, "positive": positives, "negative": negatives}

    def map(self, example: dict) -> dict:
        """Single-example transform for use with ``dataset.map``."""
        pos_passages = example["positive_passages"]
        neg_passages = example["negative_passages"]

        pos = pos_passages[self.rng.randint(0, len(pos_passages) - 1)]

        n_avail = len(neg_passages)
        if n_avail >= self.n_ways:
            sampled = self.rng.sample(neg_passages, self.n_ways)
        else:
            sampled = self.rng.choices(neg_passages, k=self.n_ways)

        neg_texts = [_passage_text(neg) for neg in sampled]

        return {
            "query": example["query"],
            "positive": _passage_text(pos),
            "negative": neg_texts[0] if self.n_ways == 1 else neg_texts,
        }
