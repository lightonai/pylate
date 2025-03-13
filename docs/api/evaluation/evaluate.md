# evaluate

Evaluate candidates matchs.



## Parameters

- **scores** (*list[list[dict]]*)

- **qrels** (*dict*)

    Qrels.

- **queries** (*list[str]*)

    index of queries of qrels.

- **metrics** (*list | None*) – defaults to `None`

    Metrics to compute.



## Examples

```python
>>> from pylate import evaluation

>>> scores = [
...     [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}],
...     [{"id": "3", "score": 0.7}, {"id": "4", "score": 0.6}],
... ]

>>> qrels = {
...     "query1": {"1": True, "2": True},
...     "query2": {"3": True, "4": True},
... }

>>> queries = ["query1", "query2"]

>>> results = evaluation.evaluate(
...     scores=scores,
...     qrels=qrels,
...     queries=queries,
...     metrics=["ndcg@10", "hits@1"],
... )
```
