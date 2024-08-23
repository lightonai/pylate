# iter_batch

Iterate over a list of elements by batch.



## Parameters

- **X** (*list[str]*)

- **batch_size** (*int*)

- **tqdm_bar** (*bool*) – defaults to `True`

- **desc** (*str*) – defaults to ``



## Examples

```python
>>> from pylate import utils

>>> X = [
...  "element 0",
...  "element 1",
...  "element 2",
...  "element 3",
...  "element 4",
... ]

>>> n_samples = 0
>>> for batch in utils.iter_batch(X, batch_size=2):
...     n_samples += len(batch)

>>> n_samples
5
```

