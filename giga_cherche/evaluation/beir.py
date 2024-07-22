"""Neural Cherche evaluation module for BEIR datasets."""

import random
from collections import defaultdict

__all__ = ["evaluate", "load_beir", "get_beir_triples"]


def add_duplicates(queries: list[str], scores: list[list[dict]]) -> list:
    """Add back duplicates scores to the set of candidates."""
    query_counts = defaultdict(int)
    for query in queries:
        query_counts[query] += 1

    query_to_result = {}
    for i, query in enumerate(iterable=queries):
        if query not in query_to_result:
            query_to_result[query] = scores[i]

    duplicated_scores = []
    for query in queries:
        if query in query_to_result:
            duplicated_scores.append(query_to_result[query])

    return duplicated_scores


def load_beir(dataset_name: str, split: str = "test") -> tuple[list, list, dict]:
    """Load BEIR dataset.

    Parameters
    ----------
    dataset_name
        Dataset name: scifact.

    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        out_dir="./evaluation_datasets/",
    )

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=split
    )

    documents = [
        {
            "id": document_id,
            "title": document["title"],
            "text": document["text"],
        }
        for document_id, document in documents.items()
    ]

    qrels = {
        queries[query_id]: query_documents
        for query_id, query_documents in qrels.items()
    }

    return documents, list(qrels.keys()), qrels


def get_beir_triples(
    key: str,
    on: list[str] | str,
    documents: list,
    queries: list[str],
    qrels: dict,
) -> list:
    """Build BEIR triples.

    Parameters
    ----------
    key
        Key.
    on
        Fields to use.
    documents
        Documents.
    queries
        Queries.

    Examples
    --------
    >>> from neural_cherche import utils

    >>> documents, queries, qrels = utils.load_beir(
    ...     "scifact",
    ...     split="test",
    ... )

    >>> triples = utils.get_beir_triples(
    ...     key="id",
    ...     on=["title", "text"],
    ...     documents=documents,
    ...     queries=queries,
    ...     qrels=qrels
    ... )

    >>> len(triples)
    339

    """
    on = on if isinstance(on, list) else [on]

    mapping_documents = {
        document[key]: " ".join([document[field] for field in on])
        for document in documents
    }

    X = []
    for query, (_, query_documents) in zip(queries, qrels.items()):
        for query_document in list(query_documents.keys()):
            # Building triples, query, positive document, random negative document
            X.append(
                (
                    query,
                    mapping_documents[query_document],
                    random.choice(seq=list(mapping_documents.values())),
                )
            )
    return X


def evaluate(
    scores: list[list[dict]],
    qrels: dict,
    queries: list[str],
    metrics: list | None = None,
) -> dict[str, float]:
    """Evaluate candidates matchs.

    Parameters
    ----------
    matchs
        Matchs.
    qrels
        Qrels.
    queries
        index of queries of qrels.
    k
        Number of documents to retrieve.
    metrics
        Metrics to compute.

    Examples
    --------
    >>> from neural_cherche import models, retrieve, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="cpu",
    ... )

    >>> documents, queries, qrels = utils.load_beir(
    ...     "scifact",
    ...     split="test",
    ... )

    >>> documents = documents[:10]

    >>> retriever = retrieve.Splade(
    ...     key="id",
    ...     on=["title", "text"],
    ...     model=model
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=1,
    ... )

    >>> documents_embeddings = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=1,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=30,
    ...     batch_size=1,
    ... )

    >>> utils.evaluate(
    ...     scores=scores,
    ...     qrels=qrels,
    ...     queries=queries,
    ...     metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
    ... )
    {'map': 0.0033333333333333335, 'ndcg@10': 0.0033333333333333335, 'ndcg@100': 0.0033333333333333335, 'recall@10': 0.0033333333333333335, 'recall@100': 0.0033333333333333335}

    """
    from ranx import Qrels, Run, evaluate

    if len(queries) > len(scores):
        scores = add_duplicates(queries=queries, scores=scores)

    qrels = Qrels(qrels=qrels)

    run_dict = {
        query: {
            match["id"]: match["similarity"]
            for rank, match in enumerate(iterable=query_matchs)
        }
        for query, query_matchs in zip(queries, scores)
    }

    run = Run(run=run_dict)

    if not metrics:
        metrics = ["ndcg@10"] + [f"hits@{k}" for k in [1, 2, 3, 4, 5, 10]]

    return evaluate(
        qrels=qrels,
        run=run,
        metrics=metrics,
        make_comparable=True,
    )
