from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def log_memory(label: str, enabled: bool = True) -> None:
    """Log current process RSS memory usage.

    Requires ``psutil`` to be installed. If not available the call is a no-op.

    Parameters
    ----------
    label
        A short description of the checkpoint being logged.
    enabled
        Whether to actually log. Callers typically pass ``self.verbose`` so
        that memory logging is gated by the same flag as other verbose output.
    """
    if not enabled:
        return
    try:
        import psutil

        rss_gb = psutil.Process().memory_info().rss / 1e9
        logger.info("[Memory] %s: %.2f GB RSS", label, rss_gb)
    except ImportError:
        pass
