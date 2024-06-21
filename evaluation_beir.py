from tqdm import tqdm

import giga_cherche.evaluation.beir as beir
from giga_cherche.indexes import WeaviateIndex
from giga_cherche.models import ColBERT
from giga_cherche.retriever import ColBERTRetriever

model = ColBERT(
    "NohTow/colbertv2_sentence_transformer",
)
WeaviateIndex = WeaviateIndex(recreate=True, max_doc_length=model.document_length)
retriever = ColBERTRetriever(WeaviateIndex)
# Input dataset for evaluation
documents, queries, qrels = beir.load_beir(
    "scifact",
    split="test",
)
batch_size = 500
i = 0
pbar = tqdm(total=len(documents))
while i < len(documents):
    end_batch = min(i + batch_size, len(documents))
    batch = documents[i:end_batch]
    documents_embeddings = model.encode(
        [doc["text"] for doc in batch],
        convert_to_numpy=False,
        is_query=False,
    )
    documents_embeddings = [
        document_embeddings.cpu().tolist()
        for document_embeddings in documents_embeddings
    ]
    doc_ids = [doc["id"] for doc in batch]
    WeaviateIndex.add_documents(
        doc_ids=doc_ids,
        doc_embeddings=documents_embeddings,
    )
    i += batch_size
    pbar.update(batch_size)

i = 0
pbar = tqdm(total=len(queries))
batch_size = 5
scores = []
while i < len(queries):
    end_batch = min(i + batch_size, len(queries))
    batch = queries[i:end_batch]
    queries_embeddings = model.encode(
        queries[i:end_batch],
        convert_to_tensor=True,
        is_query=True,
    )
    res = retriever.retrieve(queries_embeddings, 10)
    scores.extend(res)
    pbar.update(batch_size)
    i += batch_size


print(
    beir.evaluate(
        scores=scores,
        qrels=qrels,
        queries=queries,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )
)
