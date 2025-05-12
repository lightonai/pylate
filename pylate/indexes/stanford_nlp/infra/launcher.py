import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import pylate.indexes.stanford_nlp.utils.distributed as distributed
from pylate.indexes.stanford_nlp.infra.config import BaseConfig, RunConfig, RunSettings
from pylate.indexes.stanford_nlp.infra.run import Run
from pylate.indexes.stanford_nlp.utils.utils import print_message


class Launcher:
    def __init__(self, callee, run_config=None, return_all=False):
        self.callee = callee
        self.return_all = return_all

        self.run_config = RunConfig.from_existing(Run().config, run_config)
        self.nranks = self.run_config.nranks

    def launch(self, custom_config, *args):
        assert isinstance(custom_config, BaseConfig)
        assert isinstance(custom_config, RunSettings)

        return_value_queue = mp.Queue()
        rng = random.Random(time.time())
        port = str(
            12355 + rng.randint(0, 1000)
        )  # randomize the port to avoid collision on launching several jobs.
        all_procs = []
        for new_rank in range(0, self.nranks):
            new_config = type(custom_config).from_existing(
                custom_config, self.run_config, RunConfig(rank=new_rank)
            )

            args_ = (self.callee, port, return_value_queue, new_config, *args)
            all_procs.append(mp.Process(target=setup_new_process, args=args_))

        # Clear GPU space (e.g., after a `Searcher` on GPU-0 is deleted)
        # TODO: Generalize this from GPU-0 only!
        # TODO: Move this to a function. And call that function from __del__ in a class that's inherited by Searcher, Indexer, etc.

        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a

        # print_message(f"[Pre-Emptying] GPU memory check: r={r}, a={a}, f={f}")

        torch.cuda.empty_cache()

        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a

        # print_message(f"[Post-Emptying] GPU memory check: r={r}, a={a}, f={f}")

        print_memory_stats("MAIN")

        for proc in all_procs:
            print("#> Starting...")
            proc.start()

        print_memory_stats("MAIN")

        # TODO: If the processes crash upon join, raise an exception and don't block on .get() below!

        return_values = sorted([return_value_queue.get() for _ in all_procs])
        return_values = [val for rank, val in return_values]

        if not self.return_all:
            return_values = return_values[0]

        for proc in all_procs:
            proc.join()
            print("#> Joined...")

        print_memory_stats("MAIN")

        return return_values

    def launch_without_fork(self, custom_config, *args):
        assert isinstance(custom_config, BaseConfig)
        assert isinstance(custom_config, RunSettings)
        assert self.nranks == 1
        assert (
            custom_config.avoid_fork_if_possible
            or self.run_config.avoid_fork_if_possible
        )

        # new_config = type(custom_config).from_existing(
        #     custom_config, self.run_config, RunConfig(rank=0)
        # )
        new_config = custom_config
        return_val = run_process_without_mp(self.callee, new_config, *args)

        return return_val


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_process_without_mp(callee, config, *args):
    set_seed(12345)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #     map(str, config.gpus_[: config.nranks])
    # )

    with Run().context(config, inherit_config=False):
        return_val = callee(config, *args)
        torch.cuda.empty_cache()
        return return_val


def setup_new_process(callee, port, return_value_queue, config, *args):
    print_memory_stats()

    set_seed(12345)

    rank, nranks = config.rank, config.nranks

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = str(config.nranks)
    os.environ["RANK"] = str(config.rank)

    # TODO: Ideally the gpus "getter" handles this max-nranks thing!
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpus_[:nranks]))

    nranks_, distributed_ = distributed.init(rank)
    assert nranks_ == nranks

    # Run.init(args.rank, args.root, args.experiment, args.run)

    with Run().context(config, inherit_config=False):
        return_val = callee(config, *args)

    return_value_queue.put((rank, return_val))


def print_memory_stats(message=""):
    return  # FIXME: Add this back before release.

    import psutil  # Remove before releases? Or at least make optional with try/except.

    global_info = psutil.virtual_memory()
    total, available, used, free = (
        global_info.total,
        global_info.available,
        global_info.used,
        global_info.free,
    )

    info = psutil.Process().memory_info()
    rss, vms, shared = info.rss, info.vms, info.shared
    uss = psutil.Process().memory_full_info().uss

    gib = 1024**3

    summary = f"""
    "[PID: {os.getpid()}]
    [{message}]
    Available: {available / gib:,.1f} / {total / gib:,.1f}
    Free: {free / gib:,.1f} / {total / gib:,.1f}
    Usage: {used / gib:,.1f} / {total / gib:,.1f}

    RSS: {rss / gib:,.1f}
    VMS: {vms / gib:,.1f}
    USS: {uss / gib:,.1f}
    SHARED: {shared / gib:,.1f}
    """.strip().replace("\n", "\t")

    print_message(summary, pad=True)
