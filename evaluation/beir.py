"""Evaluation script for the SciFact dataset using the Beir library."""

from pylate import evaluation, indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
)

index = indexes.Voyager(override=True)

retriever = retrieve.ColBERT(index=index)

# Download the SciFact dataset
documents, queries, qrels = evaluation.load_beir(
    dataset_name="scifact",
    split="test",
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
    is_query=True,
    show_progress_bar=True,
    batch_size=32,
)

scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=10)


evaluation_scores = evaluation.evaluate(
    scores=scores,
    qrels=qrels,
    queries=queries,
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)

print(evaluation_scores)
