import os

import torch

ALREADY_INITIALIZED = False

# TODO: Consider torch.distributed.is_initialized() instead


def init(rank):
    nranks = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"])
    nranks = max(1, nranks)
    is_distributed = (nranks > 1) or ("WORLD_SIZE" in os.environ)

    global ALREADY_INITIALIZED
    if ALREADY_INITIALIZED:
        return nranks, is_distributed

    ALREADY_INITIALIZED = True

    if is_distributed and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"nranks = {nranks} \t num_gpus = {num_gpus} \t device={rank % num_gpus}")

        torch.cuda.set_device(rank % num_gpus)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return nranks, is_distributed


def barrier(rank):
    nranks = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"])
    nranks = max(1, nranks)

    if rank >= 0 and nranks > 1:
        torch.distributed.barrier(device_ids=[rank % torch.cuda.device_count()])
