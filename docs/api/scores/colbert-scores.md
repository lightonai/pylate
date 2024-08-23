# colbert_scores

Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities between the query and the document.



## Parameters

- **queries_embeddings** (*list | numpy.ndarray | torch.Tensor*)

    The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)

- **documents_embeddings** (*list | numpy.ndarray | torch.Tensor*)

    The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

- **mask** (*torch.Tensor*) – defaults to `None`



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
...     [[0.], [100.], [1.]],
...     [[1.], [0.], [1000.]],
... ])

>>> scores = colbert_scores(
...     queries_embeddings=queries_embeddings,
...     documents_embeddings=documents_embeddings
... )

>>> scores
tensor([[  10.,  100., 1000.],
        [  20.,  200., 2000.],
        [  30.,  300., 3000.]])
```

