# get_beir_triples

Build BEIR triples.



## Parameters

- **documents** (*'list'*)

    Documents.

- **queries** (*'list[str]'*)

    Queries.

- **qrels** (*'dict'*)



## Examples

```python
>>> from pylate import evaluation

>>> documents, queries, qrels = evaluation.load_beir(
...     "scifact",
...     split="test",
... )

>>> triples = evaluation.get_beir_triples(
...     documents=documents,
...     queries=queries,
...     qrels=qrels
... )

>>> len(triples)
339
```

