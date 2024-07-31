import ast

import datasets


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
    from datasets import load_dataset
    from giga_cherche import utils

    train = load_dataset(
        path="./msmarco_fr",
        name="train",
        cache_dir="./msmarco_fr"
    )

    queries = load_dataset(
        path="./msmarco_fr",
        name="queries",
        cache_dir="./msmarco_fr"
    )

    documents = load_dataset(
        path="./msmarco_fr", name="documents", cache_dir="./msmarco_fr"
    )

    train = train.map(
        utils.DatasetProcessing(
            queries=queries, documents=documents
        ).transform,
    )

    """

    def __init__(
        self, queries: datasets.Dataset, documents: datasets.Dataset, n_scores: int = 32
    ) -> None:
        self.queries = queries
        self.documents = documents
        self.n_scores = n_scores

        self.queries_index = {
            query_id: i
            for i, query_id in enumerate(iterable=self.queries["train"]["query_id"])
        }

        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(
                iterable=self.documents["train"]["document_id"]
            )
        }

    def transform(self, examples: dict) -> dict:
        """Update examples with queries and documents. Also"""
        examples["scores"] = [
            ast.literal_eval(node_or_string=score)[: self.n_scores]
            for score in examples["scores"]
        ]

        examples["query"] = [
            self.queries["train"][self.queries_index[query_id]]["text"]
            for query_id in examples["query_id"]
        ]

        n_scores = len(examples["scores"][0])

        for i in range(n_scores):
            documents = []
            for document_id in examples[f"document_id_{i}"]:
                try:
                    documents.append(
                        self.documents["train"][self.documents_index[document_id]][
                            "text"
                        ]
                    )

                except KeyError:
                    documents.append("")
            examples[f"document_{i}"] = documents

        return examples
