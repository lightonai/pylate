# KDProcessing

Dataset processing class for knowledge distillation training.



## Parameters

- **queries** (*'datasets.Dataset | datasets.DatasetDict'*)

    Queries dataset.

- **documents** (*'datasets.Dataset | datasets.DatasetDict'*)

    Documents dataset.

- **split** (*'str'*) – defaults to `train`

    Split to use for the queries and documents datasets. Used only if the queries and documents are of type `datasets.DatasetDict`.

- **n_ways** (*'int'*) – defaults to `32`

    Number of scores to keep for the distillation.



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

    - **example**     (*'dict'*)    
    
???- note "transform"

    Update the input dataset with the queries and documents.

    **Parameters**

    - **examples**     (*'dict'*)    
    
