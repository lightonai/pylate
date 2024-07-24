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
        # self.queries = {query["query_id"]: query["text"] for query in queries["train"]}
        self.queries = queries
        self.queries_index = {
            query_id: i for i, query_id in enumerate(self.queries["train"]["query_id"])
        }
        # self.documents = {
        #     document["document_id"]: document["text"] for document in documents["train"]
        # }
        self.documents = documents
        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(self.documents["train"]["document_id"])
        }

    def add_queries_and_documents(self, example: dict) -> dict:
        """Add queries and documents text to the examples."""
        scores = ast.literal_eval(node_or_string=example["scores"])

        processed_example = {
            "scores": scores,
            "query": self.queries["train"][self.queries_index[example["query_id"]]][
                "text"
            ],
        }

        n_scores = len(scores)
        for i in range(n_scores):
            try:
                processed_example[f"document_{i}"] = self.documents["train"][
                    self.documents_index[example[f"document_id_{i}"]]
                ]["text"]
            except KeyError:
                processed_example[f"document_{i}"] = ""
                print(f"KeyError: {example[f'document_id_{i}']}")
        return processed_example

    def add_queries_and_documents_transform(self, examples: dict) -> dict:
        """Add queries and documents text to the examples."""
        examples["scores"] = [
            ast.literal_eval(node_or_string=score)[:32] for score in examples["scores"]
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
                    # print("loaded")
                except KeyError:
                    documents.append("")
                    # print(f"KeyError: {document_id}")
            examples[f"document_{i}"] = documents
        # for i in range(n_scores):
        #     documents = []
        #     try:
        #         processed_example[f"document_{i}"] = self.documents["train"][
        #             self.documents_index[example[f"document_id_{i}"]]
        #         ]["text"]
        #     except KeyError:
        #         processed_example[f"document_{i}"] = ""
        #         print(f"KeyError: {example[f'document_id_{i}']}")
        return examples
