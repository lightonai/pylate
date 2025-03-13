## Retrieval evaluation

This guide demonstrates an end-to-end pipeline to evaluate the performance of the ColBERT model on retrieval tasks. The pipeline involves three key steps: indexing documents, retrieving top-k documents for a given set of queries, and evaluating the retrieval results using standard metrics.

### BEIR Retrieval Evaluation Pipeline

```python
from pylate import evaluation, indexes, models, retrieve

# Step 1: Initialize the ColBERT model

dataset = "scifact" # Choose the dataset you want to evaluate
model = models.ColBERT(
    model_name_or_path="lightonai/colbertv2.0",
    device="cuda" # "cpu" or "cuda" or "mps"
)

# Step 2: Create a Voyager index
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name=dataset,
    override=True,  # Overwrite any existing index
)

# Step 3: Load the documents, queries, and relevance judgments (qrels)
documents, queries, qrels = evaluation.load_beir(
    dataset,  # Specify the dataset (e.g., "scifact")
    split="test",  # Specify the split (e.g., "test")
)

# Step 4: Encode the documents
documents_embeddings = model.encode(
    [document["text"] for document in documents],
    batch_size=32,
    is_query=False,  # Indicate that these are documents
    show_progress_bar=True,
)

# Step 5: Add document embeddings to the index
index.add_documents(
    documents_ids=[document["id"] for document in documents],
    documents_embeddings=documents_embeddings,
)

# Step 6: Encode the queries
queries_embeddings = model.encode(
    queries,
    batch_size=32,
    is_query=True,  # Indicate that these are queries
    show_progress_bar=True,
)

# Step 7: Retrieve top-k documents
retriever = retrieve.ColBERT(index=index)
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=100,  # Retrieve the top 100 matches for each query
)

# Step 8: Evaluate the retrieval results
results = evaluation.evaluate(
    scores=scores,
    qrels=qrels,
    queries=queries,
    metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]] # NDCG for different k values
    + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]       # Hits at different k values
    + ["map"]                                         # Mean Average Precision (MAP)
    + ["recall@10", "recall@100"]                     # Recall at k
    + ["precision@10", "precision@100"],              # Precision at k
)

print(results)
```

The output is a dictionary containing various evaluation metrics. Here’s a sample output:

```python
{
    "ndcg@1": 0.47333333333333333,
    "ndcg@3": 0.543862513095773,
    "ndcg@5": 0.5623210323686343,
    "ndcg@10": 0.5891793972249917,
    "ndcg@100": 0.5891793972249917,
    "hits@1": 0.47333333333333333,
    "hits@3": 0.64,
    "hits@5": 0.7033333333333334,
    "hits@10": 0.8,
    "hits@100": 0.8,
    "map": 0.5442202380952381,
    "recall@10": 0.7160555555555556,
    "recall@100": 0.7160555555555556,
    "precision@10": 0.08,
    "precision@100": 0.008000000000000002,
}
```


???+ info
    1. is_query flag: Always set is_query=True when encoding queries and is_query=False when encoding documents. This ensures the model applies the correct prefixes for queries and documents.

    2. Evaluation metrics: The pipeline supports a wide range of evaluation metrics, including NDCG, hits, MAP, recall, and precision, with different cutoff points.

    3. Relevance judgments (qrels): The qrels are used to calculate how well the retrieved documents match the ground truth.

### BEIR datasets

The following table lists the datasets available in the BEIR benchmark along with their names, types, number of queries, corpus size, and relevance degree per query. Source: [BEIR Datasets](https://github.com/beir-cellar/beir?tab=readme-ov-file)

=== "Table"

    | Dataset       | BEIR-Name       | Type              | Queries | Corpus      |
    |---------------|-----------------|-------------------|---------|-------------|
    | MSMARCO       | msmarco          | train, dev, test  | 6,980   | 8,840,000   |
    | TREC-COVID    | trec-covid       | test              | 50      | 171,000     |
    | NFCorpus      | nfcorpus         | train, dev, test  | 323     | 3,600       |
    | BioASQ        | bioasq           | train, test       | 500     | 14,910,000  |
    | NQ            | nq               | train, test       | 3,452   | 2,680,000   |
    | HotpotQA      | hotpotqa         | train, dev, test  | 7,405   | 5,230,000   |
    | FiQA-2018     | fiqa             | train, dev, test  | 648     | 57,000      |
    | Signal-1M(RT) | signal1m         | test              | 97      | 2,860,000   |
    | TREC-NEWS     | trec-news        | test              | 57      | 595,000     |
    | Robust04      | robust04         | test              | 249     | 528,000     |
    | ArguAna       | arguana          | test              | 1,406   | 8,670       |
    | Touche-2020   | webis-touche2020 | test              | 49      | 382,000     |
    | CQADupstack   | cqadupstack      | test              | 13,145  | 457,000     |
    | Quora         | quora            | dev, test         | 10,000  | 523,000     |
    | DBPedia       | dbpedia-entity   | dev, test         | 400     | 4,630,000   |
    | SCIDOCS       | scidocs          | test              | 1,000   | 25,000      |
    | FEVER         | fever            | train, dev, test  | 6,666   | 5,420,000   |
    | Climate-FEVER | climate-fever    | test              | 1,535   | 5,420,000   |
    | SciFact       | scifact          | train, test       | 300     | 5,000       |

### Custom datasets
You can also run evaluation on your custom dataset using the following structure:

- `corpus.jsonl`: each row contains a json element with two properties: `['_id', 'text']`
    - `_id` refers to the document identifier.
    - `text` contains the text of the document.
    - (an additional `title` field can also be added if necessary)
- `queries.jsonl`: each row contains a json element with two properties: `['_id', 'text']`
    - `_id` refers to the query identifier.
    - `text` contains the text of the query.
- `qrels` folder contains tsv files with three columns: `['query-id', 'doc-id', 'score']`
    - `query-id` refers to the query identifier.
    - `doc-id` refers to the document identifier.
    - `score` contains the relation between the query and the document (1 if relevant, else 0)
The name of the tsv corresponds to the split (e.g, "dev").

You can then use the same pipeline as with BEIR datasets by changing the loading of the data in step 3:

```python
documents, queries, qrels = evaluation.load_custom_dataset(
    "custom_dataset", split="dev"
)
```


### Metrics

PyLate evaluation is based on [Ranx Python library](https://amenra.github.io/ranx/metrics/) to compute standard Information Retrieval metrics. The following metrics are supported:

=== "Table"

    | Metric                    | Alias         | @k  |
    |----------------------------|---------------|-----|
    | Hits                       | hits          | Yes |
    | Hit Rate / Success         | hit_rate      | Yes |
    | Precision                  | precision     | Yes |
    | Recall                     | recall        | Yes |
    | F1                         | f1            | Yes |
    | R-Precision                | r_precision   | No  |
    | Bpref                      | bpref         | No  |
    | Rank-biased Precision      | rbp           | No  |
    | Mean Reciprocal Rank       | mrr           | Yes |
    | Mean Average Precision     | map           | Yes |
    | DCG                        | dcg           | Yes |
    | DCG Burges                 | dcg_burges    | Yes |
    | NDCG                       | ndcg          | Yes |
    | NDCG Burges                | ndcg_burges   | Yes |


For any details about the metrics, please refer to [Ranx documentation](https://amenra.github.io/ranx/metrics/).

Sample code to evaluate the retrieval results using specific metrics:

```python
results = evaluation.evaluate(
    scores=scores,
    qrels=qrels,
    queries=queries,
    metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]] # NDCG for different k values
    + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]       # Hits at different k values
    + ["map"]                                         # Mean Average Precision (MAP)
    + ["recall@10", "recall@100"]                     # Recall at k
    + ["precision@10", "precision@100"],              # Precision at k
)
```
