# ColBERT-based Retrieval with PLAID

PyLate provides a streamlined interface to index and retrieve documents using ColBERT models, powered by our high-performance **PLAID** index.

## The PLAID Index

PyLate leverages [**PLAID**](https://arxiv.org/abs/2205.09707), a purpose-built index for fast ColBERT retrieval and specifically **[FastPLAID](https://github.com/lightonai/fast-plaid)**, an optimized implementation that delivers significant performance improvements over the original PLAID.

---

## Indexing

The following example demonstrates the end-to-end process of creating a PLAID index from a collection of documents.

### Step-by-Step Index Creation

Here's how to load a model, initialize an index, and populate it with your documents.

```python
from pylate import indexes, models

# A sample collection of documents to index
documents_ids = ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005", "doc_006", "doc_007", "doc_008"]

documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel and completed in 1889 as the entrance to the 1889 World's Fair. It is a globally recognized symbol of France and one of the most visited paid monuments in the world. The tower is 330 meters tall and has three levels for visitors.",
    "The Louvre is the world's largest art museum and a historic monument in Paris, France. It is located on the Right Bank of the Seine River in the 1st arrondissement. A central landmark of the city, the Louvre is home to some of the most famous works of art, including the Mona Lisa and the Venus de Milo. Its distinctive glass pyramid was added in 1989.",
    "ColBERT is a highly effective neural retrieval model based on late interaction. It was introduced by Omar Khattab and Matei Zaharia. Unlike traditional dense retrieval models that compute a single vector for the entire query and document, ColBERT computes a contextualized embedding for each token of the query and document, and then performs a fast, parallelized late interaction between them.",
    "Paris is known for its cafes, fashion, and the Seine River. The city's rich history dates back to the Roman era, and it has since become a major center for art, culture, and gastronomy. Landmarks like the Notre-Dame Cathedral, Arc de Triomphe, and the Sacré-Cœur Basilica add to its architectural beauty.",
    "Deep Learning based information retrieval models have advanced the state of the art in search. These models, often trained on massive datasets, can understand the nuanced semantics of language, moving beyond simple keyword matching. Techniques like Siamese networks, Transformers, and contrastive learning are commonly used to build these powerful retrieval systems.",
    "The University of Paris, known as the Sorbonne, is one of the world's oldest universities. It was founded in the mid-12th century and gained a strong reputation for academic excellence. The university has played a pivotal role in the intellectual history of Europe and has been a hub for philosophers, scientists, and writers for centuries.",
    "The Arc de Triomphe honors those who fought and died for France in the French Revolutionary and Napoleonic Wars. Located at the western end of the Champs-Élysées, it is a key landmark of Paris and a symbol of national pride. The monument stands at the center of Place Charles de Gaulle, a busy roundabout.",
    "The Seine is a major river in northern France, flowing through Paris and into the English Channel. It has been a vital waterway for commerce and a source of inspiration for countless artists. Many of Paris's famous monuments and buildings are situated along its banks, including the Notre-Dame Cathedral and the Louvre Museum."
]

# --- Step 1: Load the ColBERT Model ---
model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
)

# --- Step 2: Initialize the PLAID Index ---
index = indexes.PLAID(
    index_folder="pylate-colbert-index",
    index_name="my_documents",
    override=True,
)

# --- Step 3: Encode Documents into Embeddings ---
documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False,
    show_progress_bar=True,
)

# --- Step 4: Add Document Embeddings to the Index ---
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)

print("Indexing complete!")
```

### Persisting and Re-loading an Index

You only need to build the index once. For subsequent uses, you can load the existing index directly from disk. By setting `override=False`, you ensure that the existing index is preserved and adding documents will append to it rather than replacing it.

```python
loaded_index = indexes.PLAID(
    index_folder="pylate-colbert-index",
    index_name="my_documents",
    override=False,
)
```

---

## Retrieval

Once the index is built, you can perform searches. The process involves encoding queries and using the retriever to fetch the top-k most relevant documents.

### Step-by-Step Retrieval

This example continues from the previous section, using the index we already created.

```python
from pylate import retrieve

# Assume 'model' and 'index' are already loaded from the previous steps.

# --- Step 1: Initialize the ColBERT Retriever ---
# The retriever links the model's scoring logic with the PLAID index.
retriever = retrieve.ColBERT(index=index)

# A list of queries to search for
queries = ["monuments in Paris", "neural retrieval models"]

# --- Step 2: Encode Queries into Embeddings ---
# Encode the search queries. Note that 'is_query=True' is critical here
# as queries are processed differently from documents.
queries_embeddings = model.encode(
    queries,
    batch_size=32,
    is_query=True,
    show_progress_bar=True,
)

# --- Step 3: Retrieve Top-k Documents ---
# The 'retrieve' method searches the index and returns ranked results.
# 'k' specifies the maximum number of documents to return for each query.
search_results = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=2,
)

print(search_results)
```

**Example Output:**

```
[
    [
        {'id': 'doc_002', 'score': 28.87158203125},
        {'id': 'doc_004', 'score': 28.73193359375}
    ],
    [
        {'id': 'doc_003', 'score': 29.3173828125},
        {'id': 'doc_005', 'score': 29.087890625}
    ]
]
```

---

## Advanced Features & Optimizations

### Filtering Search Results

You can constrain a search to a specific subset of document IDs. This is useful for implementing metadata-based filtering (e.g., `date > 2024`) or searching within a user-defined collection. Filtering can also significantly accelerate searches by reducing the search space.

???+ info "Document ID Assignment"
	The document IDs used in the `subset` parameter must match the string identifiers provided in `documents_ids` during the indexing phase.

**Single filter for all queries:**

Apply the same list of allowed document IDs to every query in the batch.

```python
# Only documents "doc_001" and "doc_003" are considered for search.
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,
    subset=["doc_001", "doc_003"]
)
```

**Different filters for each query:**

Provide a list of lists, where each inner list is the specific subset of document IDs for the corresponding query.

```python
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,
    subset=[
        ["doc_001", "doc_002"],  # Filter for the first query
        ["doc_003"]             # Filter for the second query
    ]
)
```

### Compressing Document Embeddings (Pooling)

To reduce the memory and storage footprint of the index, you can "pool" similar token embeddings within a document. This technique averages token embeddings that are close to each other in the embedding space, effectively compressing the document representation.

You can enable this by setting the `pool_factor` during document encoding. A `pool_factor` of 2 will attempt to reduce the number of token embeddings by half.

???+ tip "Performance vs. Compression"
	As detailed in [this blog post](https://www.answer.ai/posts/colbert-pooling.html), a `pool_factor` of **2** can halve the index size with virtually zero drop in retrieval performance. Higher factors offer more compression at the cost of some accuracy.

    ```python
    # Example of encoding with pooling
    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,
        pool_factor=2,  # Keep 1/2 of the original tokens
        show_progress_bar=True,
    )
    ```

### Incremental Index Updates

The PLAID index computes its k-means centroids based on the initial set of documents provided. You can add new documents to an existing index at any time using `add_documents`, and `override=False` but these new documents will be assigned to the _original_ centroids.

???+ warning "Recommendation for Large-Scale Updates"
	While incremental additions are supported, adding a very large volume of new documents that significantly shifts the data distribution may lead to suboptimal clustering. For best performance after massive updates, it is recommended to rebuild the index from scratch.

---

## Performance Tuning

FastPLAID offers several parameters to fine-tune the trade-off between indexing speed, search speed, and retrieval accuracy.

### Indexing-Time Parameters

These are configured during the `indexes.PLAID` initialization.

| Parameter          | Default | Speed Impact    | Accuracy Impact          | Description                                                                           |
| :----------------- | :------ | :-------------- | :----------------------- | :------------------------------------------------------------------------------------ |
| `n_samples_kmeans` | `None`  | Lower = faster  | Lower = less precise     | Number of embeddings sampled to train k-means centroids. `None` uses all embeddings.  |
| `nbits`            | 4       | Lower = faster  | Lower = less precise     | Bits per sub-vector in Product Quantization. Controls compression level. (Range: 2-8) |
| `kmeans_niters`    | 4       | Higher = slower | Higher = better clusters | Number of iterations for the k-means clustering algorithm.                            |

**Guidelines:**

- **Fastest Indexing:** Use lower `n_samples_kmeans`, `nbits` (e.g., 2), and `kmeans_niters` (e.g., 2).
- **Highest Quality Index:** Use higher `nbits` (e.g., 8) and `kmeans_niters` (e.g., 10+).

### Search-Time Parameters

These are configured in the `retriever.retrieve` method.

| Parameter       | Default | Speed Impact    | Accuracy Impact         | Description                                                                                                        |
| :-------------- | :------ | :-------------- | :---------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `n_ivf_probe`   | 8       | Higher = slower | Higher = better recall  | Number of IVF clusters to visit for each query. A higher value increases the chance of finding relevant documents. |
| `n_full_scores` | 8192    | Higher = slower | Higher = better ranking | Number of candidate documents to re-rank with the full `MaxSim` operation.                                         |

**Guidelines:**

- **Fastest Search:** Use a lower `n_ivf_probe` (e.g., 4) and `n_full_scores` (e.g., 1024).
- **Highest Recall:** Use a higher `n_ivf_probe` (e.g., 16-32) and `n_full_scores` (e.g., 8192+).

### Stanford PLAID

Instead of using the default FastPLAID backend, you can opt for the original Stanford PLAID implementation. This is primarily for research or comparison purposes, as it is significantly slower.

```python
index = indexes.PLAID(
    index_folder="pylate-colbert-index",
    index_name="my_documents",
    override=False,
    use_fast=False,  # Use the original Stanford PLAID implementation
)
```


## Reranking

To perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use `rank.rerank` function which takes the queries and documents embeddings along with the documents ids to rerank them:

```python
from pylate import rank

queries = [
    "query A",
    "query B",
]

documents = [
    ["document A", "document B"],
    ["document 1", "document C", "document B"],
]

documents_ids = [
    [1, 2],
    [1, 3, 2],
]

queries_embeddings = model.encode(
    queries,
    is_query=True,
)

documents_embeddings = model.encode(
    documents,
    is_query=False,
)

reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
```
