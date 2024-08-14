"""Evaluation script for the miracl_fr dataset using the Beir library."""

from beir.datasets.data_loader import GenericDataLoader

from giga_cherche import evaluation, indexes, models, retrieve, utils

model = models.ColBERT(
    model_name_or_path="NohTow/colbert_xml-r-english",
    document_length=300,
)
index = indexes.Voyager(override_collection=True)
retriever = retrieve.ColBERT(index=index)

documents, queries, qrels = GenericDataLoader("datasets/miracl_fr").load(split="dev")

documents = [
    {
        "id": document_id,
        "title": document["title"],
        "text": document["text"],
    }
    for document_id, document in documents.items()
]

qrels = {
    queries[query_id]: query_documents for query_id, query_documents in qrels.items()
}
queries = list(qrels.keys())

for batch in utils.iter_batch(documents, batch_size=500):
    documents_embeddings = model.encode(
        [document["title"] + " " + document["text"] for document in batch],
        convert_to_numpy=True,
        is_query=False,
    )

    index.add_documents(
        documents_ids=[document["id"] for document in batch],
        documents_embeddings=documents_embeddings,
    )

scores = []

for batch in utils.iter_batch(queries, batch_size=5):
    queries_embeddings = model.encode(
        sentences=batch,
        convert_to_numpy=True,
        is_query=True,
    )

    scores.extend(retriever.retrieve(queries=queries_embeddings, k=10))

print(
    evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=queries,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )
)
