# load_beir

Load BEIR dataset.



## Parameters

- **dataset_name** (*str*)

    Name of the beir dataset.

- **split** (*str*) – defaults to `test`

    Split to load.



## Examples

```python
>>> from pylate import evaluation

>>> documents, queries, qrels = evaluation.load_beir(
...     "scifact",
...     split="test",
... )

>>> len(documents)
5183

>>> len(queries)
300

>>> len(qrels)
300
```

