import ast
import logging

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
    n_scores
        Number of scores to keep for the distillation.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from giga_cherche import utils

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
        self, queries: datasets.Dataset, documents: datasets.Dataset, n_ways: int = 32
    ) -> None:
        self.queries = queries["train"] if "train" in queries else queries
        self.documents = documents["train"] if "train" in documents else documents
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
        if isinstance(examples["scores"], str):
            examples["scores"] = [
                ast.literal_eval(node_or_string=score) for score in examples["scores"]
            ]

        examples["scores"] = [score[: self.n_ways] for score in examples["scores"]]

        if isinstance(examples["document_ids"], str):
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
        >>> from giga_cherche import utils

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
