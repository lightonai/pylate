# colbert_kd_scores

Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.



## Parameters

- **queries_embeddings** (*'list | np.ndarray | torch.Tensor'*)

- **documents_embeddings** (*'list | np.ndarray | torch.Tensor'*)

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
...     [[[10.], [0.], [1.]], [[20.], [0.], [1.]], [[30.], [0.], [1.]]],
...     [[[0.], [100.], [1.]], [[0.], [200.], [1.]], [[0.], [300.], [1.]]],
...     [[[1.], [0.], [1000.]], [[1.], [0.], [2000.]], [[10.], [0.], [3000.]]],
... ])
>>> documents_mask = torch.tensor([
...     [[0., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
... ])
>>> query_mask = torch.tensor([
...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
... ])
>>> colbert_kd_scores(
...     queries_embeddings=queries_embeddings,
...     documents_embeddings=documents_embeddings,
...     queries_mask=query_mask,
...     documents_mask=documents_mask,
... )
tensor([[ 1.,  20.,  30.],
        [200., 400., 600.],
        [  0.,   0.,   0.]])
```
