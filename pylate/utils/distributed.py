import torch.distributed as dist
from typing import Sequence
import torch
import logging

not_init_warning = True
logger = logging.getLogger(__name__)
def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """Collects a Tensor from each rank, and returns a list of gathered Tensors indexed by rank. The tensor for the local rank is the original one, with the gradients while the others have no gradients.
    """
    global not_init_warning
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(obj_gather_list, tensor)
        obj_gather_list[dist.get_rank()] = tensor
        return obj_gather_list
    try:
        world_size = dist.get_world_size()
        if world_size == 1:
            return [tensor]
    except ValueError:
        if not_init_warning:
            logger.warning("Trying to gather while torch.distributed is not available or has not been initialized, returning the original (local) tensor. This is expected if you are only using one GPU; consider not using gathering to remove this warning.")
            not_init_warning = False
        return [tensor]