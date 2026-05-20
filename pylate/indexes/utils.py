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


def convert_embeddings_to_torch(
    embeddings: np.ndarray | torch.Tensor | list,
) -> list[torch.Tensor]:
    """Convert embeddings to list of torch tensors as expected by fast-plaid and WARP."""
    if isinstance(embeddings, list):
        if len(embeddings) == 0:
            return []
        if isinstance(embeddings[0], torch.Tensor):
            return embeddings
        elif isinstance(embeddings[0], np.ndarray):
            return [torch.from_numpy(emb) for emb in embeddings]

    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [torch.from_numpy(embeddings[i]) for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [torch.from_numpy(embeddings)]

    if isinstance(embeddings, torch.Tensor):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [embeddings[i] for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [embeddings]

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
