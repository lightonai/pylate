# colbert_scores

Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities between the query and the document.



## Parameters

- **queries_embeddings** (*'list | np.ndarray | torch.Tensor'*)

    The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)

- **documents_embeddings** (*'list | np.ndarray | torch.Tensor'*)

    The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

- **queries_mask** (*'torch.Tensor'*) – defaults to `None`

- **documents_mask** (*'torch.Tensor'*) – defaults to `None`



## Examples

```python
>>> import torch

>>> queries_embeddings = torch.tensor([
...     [[1.], [0.], [0.], [0.]],
...     [[0.], [2.], [0.], [0.]],
...     [[0.], [0.], [3.], [0.]],
... ])

>>> documents_embeddings = torch.tensor([
...     [[10.], [0.], [1.]],
...     [[0.], [100.], [10.]],
...     [[1.], [0.], [1000.]],
... ])

>>> documents_mask = torch.tensor([
...     [1., 1., 1.],
...     [1., 0., 1.],
...     [1., 1., 1.],
... ])
>>> query_mask = torch.tensor([
...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
... ])

>>> scores = colbert_scores(
...     queries_embeddings=queries_embeddings,
...     documents_embeddings=documents_embeddings,
...     queries_mask=query_mask,
...     documents_mask=documents_mask,
... )

>>> scores
tensor([[  10.,  10., 1000.],
        [  20.,  20., 2000.],
        [  0.,  0., 0.]])
```
