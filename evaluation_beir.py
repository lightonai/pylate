"""Evaluation script for the SciFact dataset using the Beir library."""

from giga_cherche import evaluation, indexes, models, retrieve, utils

model = models.ColBERT(
    model_name_or_path="NohTow/colbertv2_sentence_transformer",
)
index = indexes.Weaviate(recreate=True, max_doc_length=model.document_length)

retriever = retrieve.ColBERT(index=index)

# Input dataset for evaluation
documents, queries, qrels = evaluation.load_beir(
    dataset_name="scifact",
    split="test",
)


for batch in utils.iter_batch(documents, batch_size=500):
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in batch],
        convert_to_numpy=True,
        is_query=False,
    )

    index.add_documents(
        doc_ids=[document["id"] for document in batch],
        doc_embeddings=documents_embeddings,
    )


scores = []
for batch in utils.iter_batch(queries, batch_size=5):
    queries_embeddings = model.encode(
        sentences=[query["text"] for query in batch],
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
