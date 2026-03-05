from __future__ import annotations

import numpy as np
import torch


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor | list,
) -> np.ndarray | list:
    """Reshape embeddings to the expected format (batch_size, n_tokens, embedding_size)."""
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    if isinstance(embeddings, torch.Tensor):
        return reshape_embeddings(embeddings=embeddings.cpu().detach().numpy())

    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        return [embedding.cpu().detach().numpy() for embedding in embeddings]

    return embeddings


def np_dtype_for(
    dtype: object,
) -> type[np.float16] | type[np.float32] | None:
    """Map a torch or numpy dtype to the corresponding numpy float type.

    Returns ``np.float16`` or ``np.float32`` for recognised dtypes,
    ``None`` otherwise.
    """
    if dtype in (torch.float16, np.float16):
        return np.float16
    if dtype in (torch.float32, np.float32):
        return np.float32
    return None
