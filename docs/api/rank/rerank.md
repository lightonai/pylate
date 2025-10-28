# rerank

Rerank the documents based on the queries embeddings.



## Parameters

- **documents_ids** (*'list[list[int | str]]'*)

    The documents ids.

- **queries_embeddings** (*'list[list[float | int] | np.ndarray | torch.Tensor]'*)

    The queries embeddings which is a dictionary of queries and their embeddings.

- **documents_embeddings** (*'list[list[float | int] | np.ndarray | torch.Tensor]'*)

    The documents embeddings which is a dictionary of documents ids and their embeddings.

- **device** (*'str'*) – defaults to `None`

    The device to use for the reranking. If None, the device of the queries embeddings will be used.



## Examples

```python
>>> from pylate import models, rank

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
... )

>>> queries = [
...     "query A",
...     "query B",
... ]

>>> documents = [
...     ["document A", "document B"],
...     ["document 1", "document C", "document B"],
... ]

>>> documents_ids = [
...    [1, 2],
...    [1, 3, 2],
... ]

>>> queries_embeddings = model.encode(
...     queries,
...     is_query=True,
...     batch_size=1,
... )

>>> documents_embeddings = model.encode(
...     documents,
...     is_query=False,
...     batch_size=1,
... )

>>> reranked_documents = rank.rerank(
...     documents_ids=documents_ids,
...     queries_embeddings=queries_embeddings,
...     documents_embeddings=documents_embeddings,
... )

>>> assert isinstance(reranked_documents, list)
>>> assert len(reranked_documents) == 2
>>> assert len(reranked_documents[0]) == 2
>>> assert len(reranked_documents[1]) == 3
>>> assert isinstance(reranked_documents[0], list)
>>> assert isinstance(reranked_documents[0][0], dict)
>>> assert "id" in reranked_documents[0][0]
>>> assert "score" in reranked_documents[0][0]
```
