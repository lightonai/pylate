# ColBERT

ColBERT retriever.



## Parameters

- **index** (*'Voyager | PLAID'*)



## Examples

```python
>>> from pylate import indexes, models, retrieve

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
...     device="cpu",
... )

>>> documents_ids = [f"document_id_{i}" for i in range(20)]
>>> documents = [f"This is the content of document {i}." for i in range(20)]

>>> documents_embeddings = model.encode(
...     sentences=documents,
...     batch_size=1,
...     is_query=False,
... )

>>> index = indexes.PLAID(
...     index_folder="test_indexes",
...     index_name="colbert",
...     override=True,
... )
✅ Index with FastPlaid backend.

>>> index = index.add_documents(
...     documents_ids=documents_ids,
...     documents_embeddings=documents_embeddings,
... )

>>> retriever = retrieve.ColBERT(index=index)

>>> queries_embeddings = model.encode(
...     ["fruits are healthy.", "fruits are good for health."],
...     batch_size=1,
...     is_query=True,
... )

>>> results = retriever.retrieve(
...     queries_embeddings=queries_embeddings,
...     k=2,
...     device="cpu",
... )

>>> assert isinstance(results, list)
>>> assert len(results) == 2

>>> queries_embeddings = model.encode(
...     "fruits are healthy.",
...     batch_size=1,
...     is_query=True,
... )

>>> results = retriever.retrieve(
...     queries_embeddings=queries_embeddings,
...     k=2,
...     device="cpu",
... )

>>> assert isinstance(results, list)
>>> assert len(results) == 1

>>> results = retriever.retrieve(
...     queries_embeddings=queries_embeddings,
...     k=2,
...     device="cpu",
...     subset=["document_id_10"],
... )
```

## Methods

???- note "retrieve"

    Retrieve documents for a list of queries.

    **Parameters**

    - **queries_embeddings**     (*'list[list | np.ndarray | torch.Tensor]'*)
    - **k**     (*'int'*)     – defaults to `10`
    - **k_token**     (*'int'*)     – defaults to `100`
    - **device**     (*'str | None'*)     – defaults to `None`
    - **batch_size**     (*'int'*)     – defaults to `50`
    - **subset**     (*'list[list[str]] | list[str] | None'*)     – defaults to `None`
