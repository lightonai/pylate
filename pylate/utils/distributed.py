from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.distributed as dist

not_init_warning = True
logger = logging.getLogger(__name__)

_has_warned_dist_not_initialized = False


def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """Gathers a tensor from each distributed rank into a list. The tensor for the local rank is the original one, with the gradients while the others have no gradients.

    - If torch.distributed is available and initialized:
      1. Creates a list of tensors (each sized like the input `tensor`).
      2. Gathers tensors from each rank into that list.
      3. Replaces the local tensor in the list with the original tensor that retains gradients.

    - If torch.distributed is either unavailable, uninitialized, or
      `world_size == 1`, it returns a list containing only the
      original tensor and throws a warning to notify the user (helpful when using a single GPU setup).

    Parameters
    ----------
    tensor:
        The input tensor to be gathered from each rank.

    Returns
    -------
    Sequence:
        A list of tensors collected from each rank. On a single GPU or when distributed is uninitialized, the list will contain only the original tensor.

    """
    global _has_warned_dist_not_initialized

    # Check if torch.distributed is properly available and initialized.
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

        # Perform all_gather.
        dist.all_gather(gathered_tensors, tensor)

        # Replace local rank's tensor with the original (retaining gradients).
        local_rank = dist.get_rank()
        gathered_tensors[local_rank] = tensor
        return gathered_tensors

    # Warn once about uninitialized or single-GPU usage.
    if not _has_warned_dist_not_initialized:
        warning = """
            Trying to gather while torch.distributed is not available or has not been initialized, 
             returning the original (local) tensor. This is expected if you are 
             only using one GPU; consider not using gathering to remove this warning.
       """
        logger.warning(warning)
        _has_warned_dist_not_initialized = True

    return [tensor]
