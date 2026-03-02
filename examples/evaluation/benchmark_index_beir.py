"""Benchmark script for BEIR datasets: reports ndcg@10 and QPS across index types."""

from __future__ import annotations

import argparse
import json
import time

import torch

from pylate import evaluation, indexes, models, retrieve

QUERY_LENGTHS = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
}

SMALL_BEIR_DATASETS = ["nfcorpus", "scifact"]


def build_index(index_type: str, dataset_name: str, model_name: str) -> indexes.Base:
    short_model = model_name.split("/")[-1]
    if index_type == "scann":
        return indexes.ScaNN(
            override=True,
            index_name=f"{dataset_name}_{short_model}",
            store_embeddings=True,
            verbose="init",
        )
    elif index_type == "plaid":
        return indexes.PLAID(
            override=True,
            index_name=f"{dataset_name}_{short_model}",
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def run_benchmark(
    model: models.ColBERT,
    dataset_name: str,
    index_type: str,
    model_name: str,
    k: int = 10,
) -> dict:
    """Run a single benchmark: encode, index, retrieve, evaluate."""
    # Load dataset
    documents, queries, qrels = evaluation.load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )
    n_docs = len(documents)
    n_queries = len(queries)

    # Encode documents
    t0 = time.perf_counter()
    documents_embeddings = model.encode(
        sentences=[doc["text"] for doc in documents],
        batch_size=2500,
        is_query=False,
        show_progress_bar=True,
    )
    doc_encode_time = time.perf_counter() - t0

    # Build index
    index = build_index(index_type, dataset_name, model_name)
    t0 = time.perf_counter()
    index.add_documents(
        documents_ids=[doc["id"] for doc in documents],
        documents_embeddings=documents_embeddings,
        batch_size=128,
    )
    index_time = time.perf_counter() - t0

    # Encode queries
    t0 = time.perf_counter()
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
    )
    query_encode_time = time.perf_counter() - t0

    # Retrieve
    retriever = retrieve.ColBERT(index=index)
    t0 = time.perf_counter()
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=k)
    retrieve_time = time.perf_counter() - t0

    qps = n_queries / retrieve_time

    # Remove self-matches (needed for some datasets)
    for (query_id, _query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    # Evaluate
    eval_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["ndcg@10", "recall@100"],
    )

    return {
        "dataset": dataset_name,
        "index_type": index_type,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "doc_encode_time": doc_encode_time,
        "query_encode_time": query_encode_time,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "qps": qps,
        **eval_scores,
    }


def print_results(results: list[dict]) -> None:
    header = f"{'Dataset':<16} {'Index':<8} {'#Docs':>7} {'#Q':>6} {'ndcg@10':>8} {'R@100':>8} {'QPS':>8} {'Retr(s)':>8} {'Idx(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<16} {r['index_type']:<8} {r['n_docs']:>7} {r['n_queries']:>6} "
            f"{r.get('ndcg@10', 0):>8.4f} {r.get('recall@100', 0):>8.4f} "
            f"{r['qps']:>8.1f} {r['retrieve_time']:>8.2f} {r['index_time']:>8.2f}"
        )
    print("=" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BEIR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SMALL_BEIR_DATASETS,
        help=f"Datasets to benchmark (default: {SMALL_BEIR_DATASETS})",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="scann",
        choices=["plaid", "scann"],
        help="Index type to benchmark (default: scann)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model to use for encoding",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write results as JSONL"
    )
    args = parser.parse_args()

    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=300,
        query_length=max(QUERY_LENGTHS.get(d, 32) for d in args.datasets),
    ).to(torch.float16)

    results = []
    for dataset in args.datasets:
        model.query_length = QUERY_LENGTHS.get(dataset, 32)
        print(f"\n>>> Benchmarking {dataset} with {args.index_type} index...")
        r = run_benchmark(model, dataset, args.index_type, args.model, k=args.k)
        results.append(r)
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(r) + "\n")

    print_results(results)
