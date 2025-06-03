from __future__ import annotations


def load_custom_dataset(path: str, split: str = "test") -> tuple[list, list, dict]:
    """Load a custom dataset.

    Parameters
    ----------
    path
        Path of the dataset.
    split
        Split to load.

    Examples
    --------
    """
    from beir.datasets.data_loader import GenericDataLoader

    documents, queries, qrels = GenericDataLoader(path).load(split=split)

    documents = [
        {
            "id": document_id,
            "text": f"{document['title']} {document['text']}".strip()
            if "title" in document
            else document["text"].strip(),
        }
        for document_id, document in documents.items()
    ]

    return documents, queries, qrels
