import ast

import datasets

__all__ = ["DatasetProcessing"]


class DatasetProcessing:
    """Preprocess the data by adding queries and documents text to the examples.

    Example:
    --------
    from datasets import load_dataset

    from giga_cherche import utils

    train = load_dataset(path="./msmarco_fr", name="train", cache_dir="./msmarco_fr")
    queries = load_dataset(path="./msmarco_fr", name="queries", cache_dir="./msmarco_fr")
    documents = load_dataset(
        path="./msmarco_fr", name="documents", cache_dir="./msmarco_fr"
    )

    train = train.map(
        utils.DatasetProcessing(
            queries=queries, documents=documents
        ).add_queries_and_documents,
        remove_columns=[feature for feature in train["train"].features if "id" in feature],
    )
    """

    def __init__(self, queries: datasets.Dataset, documents: datasets.Dataset) -> None:
        self.queries = {query["query_id"]: query["text"] for query in queries["train"]}
        self.documents = {
            document["document_id"]: document["text"] for document in documents["train"]
        }

    def add_queries_and_documents(self, example: dict) -> dict:
        """Add queries and documents text to the examples."""
        scores = ast.literal_eval(node_or_string=example["scores"])

        processed_example = {
            "scores": scores,
            "query": self.queries[example["query_id"]],
        }

        n_scores = len(scores)

        for i in range(n_scores):
            processed_example[f"document_{i}"] = self.documents[
                example[f"document_id_{i}"]
            ]

        return processed_example
