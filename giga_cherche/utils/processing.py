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
        self, queries: datasets.Dataset, documents: datasets.Dataset, n_ways: int = 32
    ) -> None:
        self.queries = queries
        self.documents = documents
        self.n_ways = n_ways

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
        """Update the input dataset with the queries and documents."""
        examples["scores"] = [
            ast.literal_eval(node_or_string=score)[: self.n_ways]
            for score in examples["scores"]
        ]
        examples["document_ids"] = [
            ast.literal_eval(node_or_string=document_ids)[: self.n_ways]
            for document_ids in examples["document_ids"]
        ]

        examples["query"] = [
            [self.queries["train"][self.queries_index[query_id]]["text"]]
            for query_id in examples["query_id"]
        ]
        examples["documents"] = []
        for doc_ids in examples["document_ids"]:
            documents = []
            for document_id in doc_ids:
                try:
                    documents.append(
                        self.documents["train"][self.documents_index[document_id]][
                            "text"
                        ]
                    )
                except KeyError:
                    documents.append("")
                    print(f"KeyError: {document_id}")
            examples["documents"].append(documents)

        return examples

    def map(self, example: dict) -> dict:
        """Add queries and documents text to the examples."""
        scores = ast.literal_eval(node_or_string=example["scores"])[: self.n_ways]
        documents_ids = ast.literal_eval(node_or_string=example["document_ids"])[
            : self.n_ways
        ]

        processed_example = {
            "scores": scores,
            "query": self.queries["train"][self.queries_index[example["query_id"]]][
                "text"
            ],
        }
        documents = []
        for document_id in documents_ids:
            try:
                documents.append(
                    self.documents["train"][self.documents_index[document_id]]["text"]
                )
            except KeyError:
                documents.append("")
                print(f"KeyError: {document_id}")
        processed_example["documents"] = documents
        return processed_example
