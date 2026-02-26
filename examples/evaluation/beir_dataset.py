"""Evaluation script for BEIR datasets with a PLAID index."""

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

    # Parse dataset_name from command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Name of the model to use (default: 'lightonai/GTE-ModernColBERT-v1')",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'fiqa')",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="plaid",
        choices=["plaid", "scann"],
        help="Type of index to use (default: 'plaid')",
    )
    parser.add_argument(
        "--retrieval_type",
        type=str,
        default="colbert",
        choices=["colbert", "xtr"],
        help="Type of retrieval to use (default: 'colbert')",
    )
    parser.add_argument(
        "--do_encode_and_index",
        action="store_true",
        help="Whether to encode and index the documents or load the index from the disk (default: False)",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    index_type = args.index_type
    model_name = args.model_name
    retrieval_type = args.retrieval_type

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    if index_type == "scann":
        index = indexes.ScaNN(
            override=args.do_encode_and_index,
            name=f"{dataset_name}_{model_name.split('/')[-1]}",
            store_embeddings=retrieval_type != "xtr",
            dimensions_per_block=1,
            anisotropic_quantization_threshold=0.1,
            verbose="all",
            index_folder="indexes",
        )

    elif index_type == "plaid":
        index = indexes.PLAID(
            override=args.do_encode_and_index,
            index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
        )

    if retrieval_type == "colbert":
        retriever = retrieve.ColBERT(index=index)
    elif retrieval_type == "xtr":
        retriever = retrieve.XTR(index=index, verbose=True)
    else:
        raise ValueError(f"Invalid retrieval type: {retrieval_type}")

    if args.do_encode_and_index:
        documents_embeddings = model.encode(
            sentences=[document["text"].lower() for document in documents],
            batch_size=1000,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        index.add_documents(
            documents_ids=[document["id"] for document in documents],
            documents_embeddings=documents_embeddings,
            batch_size=1000,
        )

    queries_embeddings = model.encode(
        sentences=[query.lower() for query in list(queries.values())],
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
        convert_to_tensor=True,
    )

    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings, k=20, k_token=40_000
    )

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                # Remove the query_id from the score
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        # queries=queries,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(evaluation_scores)
