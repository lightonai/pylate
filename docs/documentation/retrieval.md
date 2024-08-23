## ColBERT Retrieval

The ColBERT retrieval module provide a streamlined interface to index and retrieve documents using the ColBERT model. It leverages the Voyager index to efficiently handle document embeddings and enable fast retrieval.

### Indexing documents

First, initialize the ColBERT model and Voyager index, then encode and index your documents:

1. Initialize the ColBERT model.
2. Set up the Voyager index.
3. Encode documents: Ensure `is_query=False` when encoding documents so the model knows it is processing documents rather than queries.
4. Add documents to the index: Provide both document IDs and their corresponding embeddings to the index.

Here’s an example code for indexing:

```python
from pylate import indexes, models, retrieve

# Step 1: Initialize the ColBERT model
model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
)

# Step 2: Create a Voyager index
index = indexes.Voyager(
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
    is_query=False,  # Indicate that these are documents, not queries
    show_progress_bar=True,
)

# Step 4: Add document embeddings to the index
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

### Retrieving top-k documents for queries

Once the documents are indexed, you can retrieve the top-k most relevant documents for a given set of queries.

1. Initialize the ColBERT retriever
2. Encode the queries: Use the same ColBERT model. Be sure to set `is_query=True`, so the system can differentiate between queries and documents.
3. Retrieve top-k documents: Pass the query embeddings to the retriever to get the top matches, including document IDs and relevance scores.

Here’s the code for retrieving relevant documents:

```python
# Step 1: Initialize the ColBERT retriever
retriever = retrieve.ColBERT(index=index)

# Step 2: Encode the queries
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True,  # Indicate that these are queries
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

## Remove documents from the index

To remove documents from the index, use the `remove_documents` method. Provide the document IDs you want to remove from the index.

```python
index.remove_documents(["1", "2"])
```