# KDProcessing

Dataset processing class for knowledge distillation training.



## Parameters

- **queries** (*datasets.arrow_dataset.Dataset*)

    Queries dataset.

- **documents** (*datasets.arrow_dataset.Dataset*)

    Documents dataset.

- **n_ways** (*int*) – defaults to `32`



## Examples

```python
>>> from datasets import load_dataset
>>> from pylate import utils

>>> train = load_dataset(
...    path="lightonai/lighton-ms-marco-mini",
...    name="train",
...    split="train",
... )

>>> queries = load_dataset(
...    path="lightonai/lighton-ms-marco-mini",
...    name="queries",
...    split="train",
... )

>>> documents = load_dataset(
...    path="lightonai/lighton-ms-marco-mini",
...    name="documents",
...    split="train",
... )

>>> train.set_transform(
...    utils.KDProcessing(
...        queries=queries, documents=documents
...    ).transform,
... )

>>> for sample in train:
...     assert "documents" in sample and isinstance(sample["documents"], list)
...     assert "query" in sample and isinstance(sample["query"], str)
...     assert "scores" in sample and isinstance(sample["scores"], list)
```

## Methods

???- note "map"

    Process a single example.

    **Parameters**

    - **example**     (*dict*)    
    
???- note "transform"

    Update the input dataset with the queries and documents.

    **Parameters**

    - **examples**     (*dict*)    
    
