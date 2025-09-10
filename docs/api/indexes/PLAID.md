# PLAID

PLAID index with choice between fast-plaid (Rust-based) and Stanford NLP backends.

This class provides a unified interface for PLAID indexing that can use either: - FastPlaid: High-performance Rust-based implementation (default) - Stanford PLAID: Original Stanford NLP implementation (deprecated)

## Parameters

- **index_folder** (*'str'*) – defaults to `indexes`

    The folder where the index will be stored.

- **index_name** (*'str'*) – defaults to `colbert`

    The name of the index.

- **override** (*'bool'*) – defaults to `False`

    Whether to override the collection if it already exists.

- **use_fast** (*'bool'*) – defaults to `True`

    If True (default), use fast-plaid backend. If False, use Stanford PLAID backend.

- **nbits** (*'int'*) – defaults to `4`

    The number of bits to use for product quantization. Lower values mean more compression and potentially faster searches but can reduce accuracy.

- **kmeans_niters** (*'int'*) – defaults to `4`

    The number of iterations for the K-means algorithm used during index creation. This influences the quality of the initial centroid assignments.

- **max_points_per_centroid** (*'int'*) – defaults to `256`

    The maximum number of points (token embeddings) that can be assigned to a single centroid during K-means. This helps in balancing the clusters.

- **n_ivf_probe** (*'int'*) – defaults to `8`

    The number of inverted file list "probes" to perform during the search. This parameter controls the number of clusters to search within the index for each query. Higher values improve recall but increase search time.

- **n_full_scores** (*'int'*) – defaults to `8192`

    The number of candidate documents for which full (re-ranked) scores are computed. This is a crucial parameter for accuracy; higher values lead to more accurate results but increase computation.

- **n_samples_kmeans** (*'int | None'*) – defaults to `None`

    The number of samples to use for K-means clustering. If None, it defaults to a value based on the number of documents. This parameter can be adjusted to balance between speed, memory usage and clustering quality.

- **batch_size** (*'int'*) – defaults to `262144`

    The internal batch size used for processing queries. A larger batch size might improve throughput on powerful GPUs but can consume more memory.

- **show_progress** (*'bool'*) – defaults to `True`

    If set to True, a progress bar will be displayed during search operations.

- **device** (*'str | list[str] | None'*) – defaults to `None`

    Specifies the device(s) to use for computation. If None (default) and CUDA is available, it defaults to "cuda". If CUDA is not available, it defaults to "cpu". Can be a single device string (e.g., "cuda:0" or "cpu"). Can be a list of device strings (e.g., ["cuda:0", "cuda:1"]).

- **kwargs**



## Examples

```python
>>> from pylate import indexes, models

>>> index = indexes.PLAID(
...    index_folder="test_index",
...    index_name="plaid_colbert",
...    override=True,
... )
✅ Index with FastPlaid backend.

>>> model = models.ColBERT(
...    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
... )

>>> documents_embeddings = model.encode([
...    "Document content here...",
...    "Another document...",
... ] * 10, is_query=False)

>>> index = index.add_documents(
...    documents_ids=range(len(documents_embeddings)),
...    documents_embeddings=documents_embeddings
... )

>>> queries_embeddings = model.encode(
...     ["search query", "hello world"],
...     is_query=True,
... )

>>> scores = index(
...     queries_embeddings,
...     k=10,
... )

>>> index = index.add_documents(
...    documents_ids=range(len(documents_embeddings), len(documents_embeddings) * 2),
...    documents_embeddings=documents_embeddings
... )

>>> scores = index(
...     queries_embeddings,
...     k=25,
... )
```

## Methods

???- note "__call__"

    Query the index for the nearest neighbors of the query embeddings.

    **Parameters**

    - **queries_embeddings**     (*'np.ndarray | torch.Tensor'*)
    - **k**     (*'int'*)     – defaults to `10`
    - **subset**     (*'list[list[str]] | list[str] | None'*)     – defaults to `None`

    **Returns**

    *list[list[dict[str, str | float]]]*:     List of lists containing dictionaries with 'id' and 'score' keys.

???- note "add_documents"

    Add documents to the index.

    **Parameters**

    - **documents_ids**     (*'str | list[str]'*)
    - **documents_embeddings**     (*'list[np.ndarray | torch.Tensor]'*)
    - **kwargs**

???- note "get_documents_embeddings"

    Get document embeddings by their IDs.

    **Parameters**

    - **document_ids**     (*'list[list[str]]'*)

???- note "remove_documents"

    Remove documents from the index.

    **Parameters**

    - **documents_ids**     (*'list[str]'*)
