from __future__ import annotations

import logging
import queue

import torch
import torch.multiprocessing as mp
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)


def _start_multi_process_pool(model, target_devices: list[str] = None) -> dict:
    """
    Starts a multi-process pool to process the encoding with several independent processes
    via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

    This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
    to start only one process per GPU. This method works together with encode_multi_process
    and stop_multi_process_pool.

    Parameters
    ----------
    target_devices
        PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...], ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"]. If target_devices is None and CUDA/NPU
        is available, then all available CUDA/NPU devices will be used. If target_devices is None and
        CUDA/NPU is not available, then 4 CPU devices will be used.

    Returns
    -------
    A dictionary with the target processes, an input queue, and an output queue.

    """
    if target_devices is None:
        if torch.cuda.is_available():
            target_devices = [
                "cuda:{}".format(i) for i in range(torch.cuda.device_count())
            ]
        elif is_torch_npu_available():
            target_devices = [
                "npu:{}".format(i) for i in range(torch.npu.device_count())
            ]
        else:
            logger.info("CUDA/NPU is not available. Starting 4 CPU workers")
            target_devices = ["cpu"] * 4

    logger.info(
        "Start multi-process pool on devices: {}".format(
            ", ".join(map(str, target_devices))
        )
    )

    model.to("cpu")
    model.share_memory()
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    processes = []

    for device_id in target_devices:
        p = ctx.Process(
            target=_encode_multi_process_worker,
            args=(device_id, model, input_queue, output_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    return {"input": input_queue, "output": output_queue, "processes": processes}


def _encode_multi_process_worker(
    target_device: str, model, input_queue: queue.Queue, results_queue: queue.Queue
) -> None:
    """
    Internal working process to encode sentences in multi-process setup
    """
    while True:
        try:
            (
                chunk_id,
                batch_size,
                sentences,
                prompt_name,
                prompt,
                precision,
                normalize_embeddings,
                padding,
                is_query,
                pool_factor,
                protected_tokens,
            ) = input_queue.get()

            embeddings = model.encode(
                sentences,
                prompt_name=prompt_name,
                prompt=prompt,
                device=target_device,
                show_progress_bar=False,
                precision=precision,
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                padding=padding,
                is_query=is_query,
                pool_factor=pool_factor,
                protected_tokens=protected_tokens,
            )

            results_queue.put([chunk_id, embeddings])
        except queue.Empty:
            break
