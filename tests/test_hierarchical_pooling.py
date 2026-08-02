"""Tests for hierarchical / Ward token pooling."""

import numpy as np
import pytest
import torch
from scipy.cluster import hierarchy

from pylate.models.colbert import ColBERT


def _normalized_tokens(n_tokens: int = 12, dim: int = 8, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(n_tokens, dim, generator=generator)
    return torch.nn.functional.normalize(tokens, p=2, dim=1)


def test_ward_linkage_uses_euclidean_observations():
    """Ward must be built from Euclidean geometry on L2-normalized tokens."""
    tokens = _normalized_tokens()
    pooled = ColBERT.pool_embeddings_hierarchical(
        None, [tokens], pool_factor=2, protected_tokens=0
    )[0]

    expected_linkage = hierarchy.linkage(
        tokens.float().numpy(), method="ward"
    )
    labels = hierarchy.fcluster(
        expected_linkage, t=max(len(tokens) // 2, 1), criterion="maxclust"
    )
    expected = torch.stack(
        [
            tokens[labels == cluster_id].mean(dim=0)
            for cluster_id in range(1, int(labels.max()) + 1)
            if (labels == cluster_id).any()
        ]
    )

    assert pooled.shape[0] < tokens.shape[0]
    torch.testing.assert_close(pooled, expected, atol=1e-5, rtol=1e-5)


def test_pool_factor_reduces_token_count():
    tokens = _normalized_tokens(n_tokens=16)
    pooled = ColBERT.pool_embeddings_hierarchical(
        None, [tokens], pool_factor=4, protected_tokens=1
    )[0]

    assert pooled.shape[0] == 1 + max((16 - 1) // 4, 1)
    torch.testing.assert_close(pooled[0], tokens[0])


def test_error_bound_zero_keeps_all_tokens():
    tokens = _normalized_tokens()
    pooled = ColBERT.pool_embeddings_hierarchical(
        None, [tokens], protected_tokens=0, error_bound=0.0
    )[0]

    assert pooled.shape[0] == tokens.shape[0]


def test_error_bound_one_collapses_to_single_cluster():
    tokens = _normalized_tokens()
    pooled = ColBERT.pool_embeddings_hierarchical(
        None, [tokens], protected_tokens=1, error_bound=1.0
    )[0]

    assert pooled.shape[0] == 2  # protected CLS + one pooled cluster
    torch.testing.assert_close(pooled[0], tokens[0])


def test_error_bound_is_adaptive_across_documents():
    """Redundant tokens compress more than diverse tokens at the same bound."""
    generator = torch.Generator().manual_seed(1)
    base = torch.nn.functional.normalize(torch.randn(1, 8, generator=generator), dim=1)
    # Near-duplicates should merge early under a relative height cut.
    redundant = torch.nn.functional.normalize(
        base + 1e-3 * torch.randn(12, 8, generator=generator), dim=1
    )
    diverse = torch.nn.functional.normalize(
        torch.randn(12, 8, generator=generator), dim=1
    )

    pooled = ColBERT.pool_embeddings_hierarchical(
        None,
        [redundant, diverse],
        protected_tokens=0,
        error_bound=0.4,
    )

    assert pooled[0].shape[0] < pooled[1].shape[0]
    assert pooled[1].shape[0] <= diverse.shape[0]


def test_pool_factor_and_error_bound_are_mutually_exclusive():
    tokens = _normalized_tokens()
    with pytest.raises(ValueError, match="either pool_factor"):
        ColBERT.pool_embeddings_hierarchical(
            None, [tokens], pool_factor=2, error_bound=0.3
        )


def test_error_bound_must_be_unit_interval():
    tokens = _normalized_tokens()
    with pytest.raises(ValueError, match="error_bound must be in"):
        ColBERT.pool_embeddings_hierarchical(
            None, [tokens], error_bound=1.5
        )


def test_cosine_distance_ward_differs_from_euclidean_ward():
    """Sanity check that the previous cosine-distance Ward path was incorrect."""
    tokens = _normalized_tokens(n_tokens=20, dim=16, seed=7)
    observations = tokens.float().numpy()
    euclidean = hierarchy.linkage(observations, method="ward")

    cos_sim = tokens @ tokens.T
    condensed = (1 - cos_sim.numpy())[np.triu_indices(len(tokens), k=1)]
    cosine_as_ward = hierarchy.linkage(condensed, method="ward")

    assert not np.allclose(euclidean[:, 2], cosine_as_ward[:, 2])
