## ColBERT Retrieval

PyLate provides a streamlined interface to index and retrieve documents using ColBERT models.
### PLAID
This index leverages [PLAID](https://arxiv.org/abs/2205.09707) to reduce storage cost and increase speed of retrieval.

#### Indexing documents

First, load the ColBERT model and initialize the PLAID index, then encode and index your documents:

```python
from pylate import indexes, models, retrieve

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
)

# Step 2: Initialize the PLAID index
index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
    override=True,  # This overwrites the existing index if any
)

# Step 3: Encode the documents
documents_ids = ["1", "2", "3"]
documents = ["document 1 text", "document 2 text", "document 3 text"]

documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
    show_progress_bar=True,
)

# Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

???+ tip
    Note that PLAID benefits from indexing all the documents during the creation of the index (allowing to get the most accurate kmeans computation).
    Subsequent addition of documents are supported as an experimental feature.

Note that you do not have to recreate the index and encode the documents every time. Once you have created an index and added the documents, you can re-use the index later by loading it:

```python
# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
)
```

???+ tip

    #### Pooling document embeddings

    [In this blog post](https://www.answer.ai/posts/colbert-pooling.html), we showed that similar tokens in document embeddings can be pooled together to reduce the overall cost of ColBERT indexing without without losing much performance.

    You can use this feature by setting the `pool_factor` parameter when encoding the documents to only keep 1 / `pool_factor` tokens.

    The results show that using a `pool_factor` of 2 cut the memory requirement of the index in half with virtually 0 performance drop. Higher compression can be achieved at the cost of some performance, please refer to the blog post for all the details and results.

    This simple modification to the encoding call thus save a lot of space with a very contained impact on the performances:

    ```python
    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
        pool_factor=2,
        show_progress_bar=True,
    )
    ```

#### Retrieving top-k documents for queries

Once the documents are indexed, you can retrieve the top-k most relevant documents for a given set of queries.
To do so, initialize the ColBERT retriever with the index you want to search in, encode the queries and then retrieve the top-k documents to get the top matches ids and relevance scores:

```python
# Step 1: Initialize the ColBERT retriever
retriever = retrieve.ColBERT(index=index)

# Step 2: Encode the queries
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
    show_progress_bar=True,
)

# Step 3: Retrieve top-k documents
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,  # Retrieve the top 10 matches for each query
)

print(scores)
```

Example output

```python
[
    [   # Candidates for the first query
        {"id": "3", "score": 11.266985893249512},
        {"id": "1", "score": 10.303335189819336},
        {"id": "2", "score": 9.502392768859863},
    ],
    [   # Candidates for the second query
        {"id": "1", "score": 10.88800048828125},
        {"id": "3", "score": 9.950843811035156},
        {"id": "2", "score": 9.602447509765625},
    ],
]
```

#### Remove documents from the index

To remove documents from the index, use the `remove_documents` method. Provide the document IDs you want to remove from the index:

```python
index.remove_documents(["1", "2"])
```

#### Parameters affecting the retrieval performance

The retrieval is not an exact search, which mean that certain parameters can affect the quality of the approximate search:

- `nbits`: the number of bits to use for the quantization of the residual. Usually set to 2, larger values (4-8) will decrease the quantization errors but create larger indexes.
- `kmeans_niters`: the number of iterations to use for the k-means clustering. Larger values will create better centroids but takes longer to create (and default value is often more than enough to converge)
- `ndocs`: the number of candidate documents to score. Larger values will make querying slower and use more memory, but might results in better results.
- `centroid_score_threshold`: the threshold scores for centroid pruning. Larger values will consider more candidate centroids and so can find better neighbors but will be slower.
- `ncells`: the maximum numbers of cells to keep during search. Larger values will consider more centroids and more candidate but will be slower.

### Voyager index

We also provide an index leveraging the Voyager HNSW index to efficiently handle document embeddings and enable fast retrieval. This was the default index before we implemented PLAID and suggest using PLAID if possible.
To use it, simply replace `PLAID` by `Voyager`:

```python
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
    override=True,  # This overwrites the existing index if any
)
```

#### Parameters affecting the retrieval performance

The retrieval is not an exact search, which mean that certain parameters can affect the quality of the approximate search:

- `M`: the maximum number of connections of a node in the graph. Higher values will improve recall and reduce retrieval time but will increase memory usage and the creation time of the index.
- `ef_construction`: the maximum number of neighbors for a node during the creation of the index. Higher values increase the quality of the index but increase the creation time of the index.
- `ef_search`: the maximum number of neighbors for a node during the search. Higher values increase the quality of the search but also the search time.

Refer to [HNSW documentation for more details](https://www.pinecone.io/learn/series/faiss/hnsw/).

???+ info
    Another parameter that significantly influences search quality is **k_token**. This parameter determines the **number of neighbors retrieved for each query token**. Higher values of k_token will consider more candidates, leading to better results but at the cost of slower search performance.

    ```python
    index = indexes.Voyager(
        index_folder="pylate-index",
        index_name="index",
        override=True,  # This overwrites the existing index if any
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )

    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=10,  # Retrieve the top 10 matches for each query
        k_token=200 # retrieve 200 candidates per query token
    )
    ```


## ColBERT reranking

If you only want to use the ColBERT model to perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use rank function and pass the queries and documents to rerank:

```python
from pylate import rank, models

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

model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
)

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

Sample output:

```
[
    [
        {"id": 1, "score": 13.866096496582031},
        {"id": 2, "score": 7.363473415374756}
    ],
    [
        {"id": 2, "score": 16.025644302368164},
        {"id": 3, "score": 7.144075870513916},
        {"id": 1, "score": 4.203659534454346},
    ],
]
```
