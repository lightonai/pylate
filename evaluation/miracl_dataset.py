"""Evaluation script for the miracl_fr dataset using the Beir library."""

from beir.datasets.data_loader import GenericDataLoader

from pylate import evaluation, indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="lightonai/colbertv2.0",
    document_length=300,
)
index = indexes.Voyager(override=True)
retriever = retrieve.ColBERT(index=index)

documents, queries, qrels = GenericDataLoader("datasets/miracl_fr").load(split="dev")

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
    queries[query_id]: query_documents for query_id, query_documents in qrels.items()
}
queries = list(qrels.keys())


documents_embeddings = model.encode(
    sentences=[document["text"] + " " + document["title"] for document in documents],
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
