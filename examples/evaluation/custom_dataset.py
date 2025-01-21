"""Evaluation script for the custom dataset."""

from __future__ import annotations

from pylate import evaluation, indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="lightonai/colbertv2.0",
    document_length=300,
)
index = indexes.Voyager(override=True)
retriever = retrieve.ColBERT(index=index)

documents, queries, qrels = evaluation.load_custom_dataset(
    "datasets/miracl_fr", split="dev"
)

documents_embeddings = model.encode(
    sentences=[document["text"] for document in documents],
    batch_size=32,
    is_query=False,
    show_progress_bar=True,
)

index.add_documents(
    documents_ids=[document["id"] for document in documents],
    documents_embeddings=documents_embeddings,
)

queries_embeddings = model.encode(
    sentences=queries,
    batch_size=32,
    is_query=True,
    show_progress_bar=True,
)

scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)


evaluation_scores = evaluation.evaluate(
    scores=scores,
    qrels=qrels,
    queries=queries,
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)

print(evaluation_scores)
