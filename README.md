# giga-cherche

giga-cherche is a library based on [sentence-transformers](https://github.com/UKPLab/sentence-transformers) to train and use ColBERT models.

# Installation

giga-cherche can be installed by running the setup.py file with the needed extras from the following list:
- ```index``` if you want to use the proposed indexes
- ```eval``` if you need to run BEIR evaluations
- ```dev``` if you want to contribute to the repository
  
For example, to run the BEIR evaluations using giga-cherche indexes:
```python setup.py install --extras eval, index```

# Modeling
The modeling of giga-cherche is based on sentence-transformers which allow to build a ColBERT model from any encoder available by appending a projection layer applied to the output of the encoders to reduce the embeddings dimension. 
```
from giga_cherche.models import ColBERT
model_name = "bert-base-uncased"
model = ColBERT(model_name_or_path=model_name)
```
The following parameters can be passed to the constructor to set different properties of the model:
- ```embedding_size```, the output size of the projection layer and so the dimension of the embeddings
- ```query_prefix```, the string version of the query marker to be prepended when encoding queries
- ```document_prefix```, the string version of the document marker to be prepended when encoding documents
- ```query_length```, the length of the query to truncate / pad to with mask tokens
- ```document_length```, the length of the document to truncate
- ```attend_to_expansion_tokens```, whether queries tokens should attend to MASK expansion tokens (original ColBERT did not)
- ```skiplist_words```, a list of words to ignore in documents during scoring (default to punctuation)

## Training

Given that giga-cherche ColBERT models are sentence-transformers models, we can benefit from all the bells and whistles from the latest update, including multi-gpu and BF16 training.
For now, you can train ColBERT models using triplets dataset (datasets containing a positive and a negative for each query). The syntax is the same as sentence-transformers, using the specific elements adapted to ColBERT from giga-cherche:

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from giga_cherche import losses, models, data_collator, evaluation

model_name = "bert-base-uncased"
batch_size = 32
num_train_epochs = 1
output_dir = "colbert_base"

model = models.ColBERT(model_name_or_path=model_name)

dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
splits = dataset.train_test_split(test_size=0.1)
train_dataset = splits["train"]
eval_dataset = splits["test"]

train_loss = losses.ColBERT(model=model)

dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    learning_rate=3e-6,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=data_collator.ColBERT(model.tokenize),
)

trainer.train()
```

## Tokenization

```
import ast 

def add_queries_and_documents(example: dict) -> dict:
    """Add queries and documents text to the examples."""
    scores = ast.literal_eval(node_or_string=example["scores"])
    processed_example = {"scores": scores, "query": queries[example["query_id"]]}

    n_scores = len(scores)
    for i in range(n_scores):
        processed_example[f"document_{i}"] = documents[example[f"document_id_{i}"]]
    
    return processed_example
```

##  Inference
Once trained, the model can then be loaded to perform inference (you can also load the models directly from Hugging Face, for example using the provided ColBERTv2 model [NohTow/colbertv2_sentence_transformer](https://huggingface.co/NohTow/colbertv2_sentence_transformer)):

```python
model = ColBERT(
    "NohTow/colbertv2_sentence_transformer",
)
```

You can then call the ```encode``` function to get the embeddings corresponding to your queries:

```python
queries_embeddings = model.encode(
        ["Who is the president of the USA?", "When was the last president of the USA elected?"],
    )
```

When encoding documents, simply set the ```is_query``` parameter to false:

```python
documents_embeddings = model.encode(
        ["Joseph Robinette Biden Jr. is an American politician who is the 46th and current president of the United States since 2021. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama and represented Delaware in the United States Senate from 1973 to 2009.", "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021."],
        is_query=False,
    )
```

By default, this will return a list of numpy arrays containing the different embeddings of each sequence in the batch. You can pass the argument ```convert_to_tensor=True``` to get a list of tensors.

We also provide the option to pool the document embeddings using hierarchical clustering. Our recent study showed that we can pool the document embeddings by a factor of 2 to halve the memory consumption of the embeddings without degrading performance. This is done by feeding ```pool_factor=2```to the encode function. Bigger pooling values can be used to obtain different size/performance trade-offs.
Note that query embeddings cannot be pooled.

You can then compute the ColBERT max-sim scores like this:

```python
from giga_cherche import scores
similarity_scores = scores.colbert_score(query_embeddings, document_embeddings)
```

## Indexing

We provide a ColBERT index based on the [Weaviate vectordb](https://weaviate.io/). To speed-up the processing, the latest async client is used and the document candidates are generated using an HNSW index, which replace the IVF index from the original ColBERT. 

Before being able to create and use an index, you need to need to launch the Weaviate server using Docker (```docker compose up```).

To populate an index, simply create it and then add the computed embeddings with their corresponding ids:

```python
from giga_cherche import indexes

index = indexes.Weaviate(name="test_index")

documents_embeddings = model.encode(
    ["Document text 1", "Document text 2"],
    is_query=False,
)

index.add_documents(
    doc_ids=["1", "2"],
    doc_embeddings=documents_embeddings,
)
```

We can also remove documents from the index using their ids:

```python
index.remove_documents(["1"])
```

To retrieve documents from the index, you can use the following code snippet:

```python
from giga_cherche import retrieve

retriever = retrieve.ColBERT(Weaviate)

queries_embeddings = model.encode(
    ["A query related to the documents", "Another query"],
)

retrieved_chunks = retriever.retrieve(queries_embeddings, k=10)
```

You can also simply rerank a list of ids produced by an upstream retrieval module (such as BM25):

```python
from giga_cherche import rerank

reranker = rerank.ColBERT(Weaviate)

reranked_chunks = reranker.rerank(
    queries_embeddings, batch_doc_ids=[["7912", "4983"], ["8726", "7891"]]
)
```

## Evaluation

We can eavaluate the performance of the model using the BEIR evaluation framework. The following code snippet shows how to evaluate the model on the SciFact dataset:

```python
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
```