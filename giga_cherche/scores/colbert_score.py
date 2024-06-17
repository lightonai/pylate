import logging
from typing import Union

import numpy as np
import torch
from sentence_transformers.util import _convert_to_batch_tensor
from torch import Tensor

logger = logging.getLogger(__name__)


def colbert_score(
    a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor], mask: Tensor
) -> Tensor:
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
    simis = torch.einsum("ash,bth->abst", a, b)
    # Expanding mask from (num_doc, doc_len) to (num_query, num_doc, query_len, doc_len)
    expanded_mask = mask.unsqueeze(0).unsqueeze(2)
    expanded_mask = expanded_mask.expand(simis.size(0), -1, simis.size(2), -1)
    # Masking out the padding tokens
    simis[expanded_mask == 0] = 0
    return simis.max(axis=3).values.sum(axis=2)
    return torch.einsum("ash,bth->abst", a, b).max(axis=3).values.sum(axis=2)


# TODO: only compute the diagonal
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
