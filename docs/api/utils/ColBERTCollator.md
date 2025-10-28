# ColBERTCollator

Collator for ColBERT model.



## Parameters

- **tokenize_fn** (*'Callable'*)

    The function to tokenize the input text.

- **valid_label_columns** (*'list[str] | None'*) – defaults to `None`

    The name of the columns that contain the labels: scores or labels.



## Examples

```python
>>> from pylate import models, utils

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
... )

>>> collator = utils.ColBERTCollator(
...     tokenize_fn=model.tokenize,
... )

>>> features = [
...     {
...         "query": "fruits are healthy.",
...         "positive": "fruits are good for health.",
...         "negative": "fruits are bad for health.",
...         "label": [0.7, 0.3]
...     }
... ]

>>> features = collator(features=features)

>>> fields = [
...     "query_input_ids",
...     "positive_input_ids",
...     "negative_input_ids",
...     "query_attention_mask",
...     "positive_attention_mask",
...     "negative_attention_mask",
...     "query_token_type_ids",
...     "positive_token_type_ids",
...     "negative_token_type_ids",
... ]

>>> for field in fields:
...     assert field in features
...     assert isinstance(features[field], torch.Tensor)
...     assert features[field].ndim == 2
```

## Methods

???- note "__call__"

    Collate a list of features into a batch.

    **Parameters**

    - **features**     (*'list[dict]'*)
