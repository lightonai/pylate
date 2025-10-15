# convert_to_tensor

Converts a list or numpy array to a torch tensor.



## Parameters

- **x** (*'torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray | list | float]'*)

    The input data. It can be a torch tensor, a numpy array, or a list of torch tensors, numpy arrays, or lists.



## Examples

```python
>>> import numpy as np
>>> import torch

>>> x = torch.tensor([[1., 1., 1.], [2., 2., 2.]])
>>> convert_to_tensor(x)
tensor([[1., 1., 1.],
        [2., 2., 2.]])

>>> x = np.array([[1., 1., 1.], [2., 2., 2.]], dtype=np.float32)
>>> convert_to_tensor(x)
tensor([[1., 1., 1.],
        [2., 2., 2.]])

>>> x = []
>>> convert_to_tensor(x)
tensor([])

>>> x = [np.array([1., 1., 1.])]
>>> convert_to_tensor(x)
tensor([[1., 1., 1.]])

>>> x = [[1., 1., 1.]]
>>> convert_to_tensor(x)
tensor([[1., 1., 1.]])

>>> x = [torch.tensor([1., 1., 1.]), torch.tensor([2., 2., 2.])]
>>> convert_to_tensor(x)
tensor([[1., 1., 1.],
        [2., 2., 2.]])

>>> x = np.array([], dtype=np.float32)
>>> convert_to_tensor(x)
tensor([])
```

