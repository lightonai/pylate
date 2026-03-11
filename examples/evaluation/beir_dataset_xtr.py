"""Evaluation script for BEIR datasets with ColBERT or XTR retrieval."""

from __future__ import annotations

import argparse

from pylate import evaluation, indexes, models, retrieve

if __name__ == "__main__":
    query_len = {
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
        "cqadupstack/android": 32,
        "cqadupstack/english": 32,
        "cqadupstack/gaming": 32,
        "cqadupstack/gis": 32,
        "cqadupstack/mathematica": 32,
        "cqadupstack/physics": 32,
        "cqadupstack/programmers": 32,
        "cqadupstack/stats": 32,
        "cqadupstack/tex": 32,
        "cqadupstack/unix": 32,
        "cqadupstack/webmasters": 32,
        "cqadupstack/wordpress": 32,
    }

    parser = argparse.ArgumentParser(
        description="Evaluate ColBERT or XTR retrieval on BEIR datasets."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the BEIR dataset (default: nfcorpus)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        choices=["colbert", "xtr"],
        default="colbert",
        help="Retrieval strategy: 'colbert' (full reranking) or 'xtr' (token-level imputation)",
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=["plaid", "scann", "voyager"],
        default=None,
        help="Index backend. Defaults to 'plaid' for colbert, 'scann' for xtr.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of documents to retrieve (default: 20)",
    )
    parser.add_argument(
        "--k_token",
        type=int,
        default=None,
        help="Token-level candidates per query token. "
        "Defaults to 100 for colbert, 10000 for xtr.",
    )
    parser.add_argument(
        "--imputation",
        type=str,
        default="min",
        choices=["min", "zero", "mean", "percentile", "power_law"],
        help="XTR imputation strategy (default: min). Only used with --retrieval xtr.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Batch size for document encoding (default: 2000)",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Batch size for query encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for retrieval scoring (e.g. 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--document_length",
        type=int,
        default=300,
        help="Document length for the model (default: 300)",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase all documents and queries before encoding.",
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name

    # Resolve defaults that depend on retrieval strategy.
    index_type = args.index or ("plaid" if args.retrieval == "colbert" else "scann")
    k_token = args.k_token or (100 if args.retrieval == "colbert" else 10_000)

    if args.retrieval == "xtr" and index_type == "plaid":
        parser.error("XTR retrieval does not support the PLAID index.")

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=args.document_length,
        query_length=query_len.get(dataset_name),
    )

    # Load dataset.
    if "cqadupstack" in dataset_name:
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        index_dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )
        index_dataset_name = dataset_name

    # Optionally lowercase all text.
    if args.lowercase:
        for doc in documents:
            doc["text"] = doc["text"].lower()
        queries = {qid: text.lower() for qid, text in queries.items()}

    # Build index.
    index_name = f"{index_dataset_name}_{model_name.split('/')[-1]}_{args.retrieval}"
    if index_type == "plaid":
        index = indexes.PLAID(override=True, index_name=index_name)
    elif index_type == "scann":
        index = indexes.ScaNN(
            override=True,
            index_name=index_name,
            store_embeddings=False,
        )
    else:
        index = indexes.Voyager(override=True, index_name=index_name)

    # Encode and index documents.
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    # Encode queries.
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=args.query_batch_size,
    )

    # Retrieve.
    retrieve_kwargs = dict(
        queries_embeddings=queries_embeddings,
        k=args.k,
        k_token=k_token,
    )
    if args.device is not None:
        retrieve_kwargs["device"] = args.device

    if args.retrieval == "colbert":
        retriever = retrieve.ColBERT(index=index)
        scores = retriever.retrieve(**retrieve_kwargs)
    else:
        retriever = retrieve.XTR(index=index, verbose=True)
        retrieve_kwargs["imputation"] = args.imputation
        scores = retriever.retrieve(**retrieve_kwargs)

    # Remove self-matches (needed for e.g. FiQA).
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(evaluation_scores)
