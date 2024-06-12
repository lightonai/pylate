import functools
import heapq
import importlib
import logging
import os
import queue
import sys
from contextlib import contextmanager
from typing import Callable, Dict, List, Literal, Optional, Union, overload

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import Tensor, device
from tqdm.autonotebook import tqdm
from transformers import is_torch_npu_available

from sentence_transformers.util import _convert_to_tensor, _convert_to_batch_tensor

logger = logging.getLogger(__name__)


def colbert_score(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Computes the ColBERT score for all pairs of vectors in a and b.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = colbert_score(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return torch.einsum("ash,bth->abst", a, b).max(axis=3).values.sum(axis=2)


#TODO: only compute the diagonal
def colbert_pairwise_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise ColBERT score colbert_score(a[i], b[i]).
    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return torch.einsum("ash,bth->abst", a, b).max(axis=3).values.sum(axis=2).diag()


# def dot_score(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
#     """
#     Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

#     Args:
#         a (Union[list, np.ndarray, Tensor]): The first tensor.
#         b (Union[list, np.ndarray, Tensor]): The second tensor.

#     Returns:
#         Tensor: Matrix with res[i][j] = dot_prod(a[i], b[j])
#     """
#     a = _convert_to_batch_tensor(a)
#     b = _convert_to_batch_tensor(b)

#     return torch.mm(a, b.transpose(0, 1))


# def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
#     """
#     Computes the pairwise dot-product dot_prod(a[i], b[i]).

#     Args:
#         a (Union[list, np.ndarray, Tensor]): The first tensor.
#         b (Union[list, np.ndarray, Tensor]): The second tensor.

#     Returns:
#         Tensor: Vector with res[i] = dot_prod(a[i], b[i])
#     """
#     a = _convert_to_tensor(a)
#     b = _convert_to_tensor(b)

#     return (a * b).sum(dim=-1)