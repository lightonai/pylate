import numpy as np
import torch


def convert_to_tensor(
    x: torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray | list | float],
) -> torch.Tensor:
    """Converts a list or numpy array to a torch tensor.

    Parameters
    ----------
    x
        The input data. It can be a torch tensor, a numpy array, or a list of torch tensors, numpy arrays, or lists.

    Examples
    --------
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

    """
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    if isinstance(x, list):
        if not x:
            return torch.tensor([], dtype=torch.float32)

        if isinstance(x[0], np.ndarray):
            return torch.from_numpy(np.array(x, dtype=np.float32))

        if isinstance(x[0], list):
            return torch.tensor(x, dtype=torch.float32)

        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
