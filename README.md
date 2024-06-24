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

# Training

Given that giga-cherche ColBERT models are sentence-transformers models, we can benefit from all the bells and whistles from the latest update, including multi-gpu and BF16 training.
For now, you can train ColBERT models using triplets dataset (datasets containing a positive and a negative for each query). The syntax is the same as sentence-transformers, using the specific elements adapted to ColBERT from giga-cherche:
```
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from giga_cherche.data_collator import ColBERTDataCollator
from giga_cherche.evaluation import ColBERTTripletEvaluator
from giga_cherche.losses import ColBERTLoss
from giga_cherche.models import ColBERT

model_name = "bert-base-uncased"
batch_size = 32
num_train_epochs = 1
output_dir = "colbert_base"

model = ColBERT(model_name_or_path=model_name)

dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
splits = dataset.train_test_split(test_size=0.1)
train_dataset = splits["train"]
eval_dataset = splits["test"]

train_loss = ColBERTLoss(model=model)
dev_evaluator = ColBERTTripletEvaluator(
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
    data_collator=ColBERTDataCollator(model.tokenize),
)

trainer.train()
```

# Inference
Once trained, the model can then be loaded to perform inference (you can also load the models directly from Hugging Face, for example using the provided ColBERTv2 model [NohTow/colbertv2_sentence_transformer](https://huggingface.co/NohTow/colbertv2_sentence_transformer)):

```
model = ColBERT(
    "NohTow/colbertv2_sentence_transformer",
)
```
You can then call the ```encode``` function to get the embeddings corresponding to your queries:
```
queries_embeddings = model.encode(
        ["Who is the president of the USA?", "When was the last president of the USA elected?"],
    )
```
When encoding documents, simply set the ```is_query``` parameter to false:
```
documents_embeddings = model.encode(
        ["Joseph Robinette Biden Jr. is an American politician who is the 46th and current president of the United States since 2021. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama and represented Delaware in the United States Senate from 1973 to 2009.", "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021."],
        is_query=False,
    )
```
By default, this will return a list of numpy arrays containing the different embeddings of each sequence in the batch. You can pass the argument ```convert_to_tensor=True``` to get a list of tensors.

We also provide the option to pool the document embeddings using hierarchical clustering. Our recent study showed that we can pool the document embeddings by a factor of 2 to halve the memory consumption of the embeddings without degrading performance. This is done by feeding ```pool_factor=2```to the encode function. Bigger pooling values can be used to obtain different size/performance trade-offs.
Note that query embeddings cannot be pooled.

You can then compute the ColBERT max-sim scores like this:
```
from giga_cherche.scores.colbert_score import colbert_score
similarity_scores = colbert_score(query_embeddings, document_embeddings)
```

# Indexing

We provide a ColBERT index based on the [Weaviate vectordb](https://weaviate.io/). To speed-up the processing, the latest async client is used and the document candidates are generated using an HNSW index, which replace the IVF index from the original ColBERT.

To populate an index, simply create it and then add the computed embeddings with their corresponding ids:
````
WeaviateIndex = WeaviateIndex(name="test_index")
documents_embeddings = model.encode(
    ["Document text 1", "Document text 2"],
    is_query=False,
)
WeaviateIndex.add_documents(
    doc_ids=["1", "2"],
    doc_embeddings=documents_embeddings,
)
```
You can then remove documents based on their ids:
```
WeaviateIndex.remove_documents(["1"])
```

You can then search into the documents of your index using a retrieval object:
```
from giga_cherche.retriever import ColBERTRetriever
queries_embeddings = model.encode(
    ["My query"],
)

retriever = ColBERTRetriever(WeaviateIndex)
retrieved_chunks = retriever.retrieve(queries_embeddings, k=10)
```
You can also simply rerank a list of ids produced by an upstream retrieval module (such as BM25):

```
from giga_cherche.reranker import ColBERTReranker
reranker = ColBERTReranker(WeaviateIndex)
reranked_chunks = reranker.rerank(
    queries_embeddings, batch_doc_ids=[["7912", "4983"], ["8726", "7891"]]
)
```

# BEIR evaluation

You can evaluate your ColBERT model on BEIR by indexing the corresponding dataset and then performing retrieval:
```
from tqdm import tqdm
import giga_cherche.evaluation.beir as beir
from giga_cherche.indexes import WeaviateIndex
from giga_cherche.models import ColBERT
from giga_cherche.retriever import ColBERTRetriever

dataset = "scifact"
model = ColBERT(
    "NohTow/colbertv2_sentence_transformer",
)
WeaviateIndex = WeaviateIndex(name=dataset, recreate=True)
retriever = ColBERTRetriever(WeaviateIndex)
# Input dataset for evaluation
documents, queries, qrels = beir.load_beir(
    dataset,
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
        is_query=False,
    )
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
```