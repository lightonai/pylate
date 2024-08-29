def load_custom_dataset(path: str, split: str = "test") -> tuple[list, list, dict]:
    from beir.datasets.data_loader import GenericDataLoader

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

    qrels = {
        queries[query_id]: query_documents
        for query_id, query_documents in qrels.items()
    }

    return documents, list(qrels.keys()), qrels
