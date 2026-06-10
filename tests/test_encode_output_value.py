"""Non-regression tests for ColBERT.encode output_value parameter.

The ``output_value=None`` mode (added for TachiomIndex) must not change the
default ``output_value="token_embeddings"`` behaviour, and the dict it returns
must stay aligned with the default output once the masks are applied.
"""

import numpy as np
import pytest

from pylate import models

DOCUMENTS = [
    "Document about apples and their nutritional benefits.",
    "Document about bananas and their vitamin content.",
    "Document about cherries and antioxidants.",
]

QUERIES = ["fruits and nutrition", "berries and immune support"]


@pytest.fixture(scope="module")
def model():
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )


def test_encode_output_value_none_documents(model):
    """The dict output, once masked, must match the default document embeddings."""
    default_embeddings = model.encode(DOCUMENTS, is_query=False)
    encoded = model.encode(DOCUMENTS, is_query=False, output_value=None)

    assert isinstance(encoded, dict)
    assert encoded.keys() == {
        "token_embeddings",
        "input_ids",
        "attention_mask",
        "masks",
    }
    assert all(len(values) == len(DOCUMENTS) for values in encoded.values())

    for default, emb, ids, mask in zip(
        default_embeddings,
        encoded["token_embeddings"],
        encoded["input_ids"],
        encoded["masks"],
    ):
        # Embeddings, token IDs and masks are aligned per (unfiltered) token.
        assert emb.shape[0] == ids.shape[0] == mask.shape[0]
        # Applying the mask reproduces the default (filtered) embeddings.
        np.testing.assert_allclose(emb[mask], default, atol=1e-6)
        # Masked token IDs are aligned 1-to-1 with the filtered embeddings.
        assert ids[mask].shape[0] == default.shape[0]


def test_encode_output_value_none_queries(model):
    """The dict output, once masked, must match the default query embeddings."""
    default_embeddings = model.encode(QUERIES, is_query=True)
    encoded = model.encode(QUERIES, is_query=True, output_value=None)

    for default, emb, ids, mask in zip(
        default_embeddings,
        encoded["token_embeddings"],
        encoded["input_ids"],
        encoded["masks"],
    ):
        assert emb.shape[0] == ids.shape[0] == mask.shape[0]
        np.testing.assert_allclose(emb[mask], default, atol=1e-6)


def test_encode_output_value_default_is_unchanged(model):
    """Passing output_value='token_embeddings' explicitly must equal the default call."""
    implicit = model.encode(DOCUMENTS, is_query=False)
    explicit = model.encode(DOCUMENTS, is_query=False, output_value="token_embeddings")

    assert len(implicit) == len(explicit)
    for implicit_embedding, explicit_embedding in zip(implicit, explicit):
        np.testing.assert_allclose(implicit_embedding, explicit_embedding, atol=1e-6)


def test_encode_output_value_none_single_string(model):
    """A single string input returns a dict of single items, not lists of one."""
    encoded = model.encode(DOCUMENTS[0], is_query=False, output_value=None)

    assert isinstance(encoded, dict)
    assert encoded["token_embeddings"].ndim == 2
    assert (
        encoded["token_embeddings"].shape[0]
        == encoded["input_ids"].shape[0]
        == encoded["masks"].shape[0]
    )

    default_embedding = model.encode(DOCUMENTS[0], is_query=False)
    np.testing.assert_allclose(
        encoded["token_embeddings"][encoded["masks"]], default_embedding, atol=1e-6
    )


def test_encode_output_value_none_with_padding(model):
    """With padding=True, all dict entries are padded to the same length."""
    encoded = model.encode(DOCUMENTS, is_query=False, output_value=None, padding=True)

    lengths = {value.shape[1] for values in encoded.values() for value in values}
    assert len(lengths) == 1, "All entries should be padded to the same length"


def test_encode_output_value_none_rejects_pool_factor(model):
    """Pooling merges tokens, so token IDs can no longer align with embeddings."""
    with pytest.raises(ValueError, match="pool_factor"):
        model.encode(DOCUMENTS, is_query=False, output_value=None, pool_factor=2)


def test_encode_output_value_invalid_raises(model):
    """Only 'token_embeddings' and None are valid output_value arguments."""
    with pytest.raises(ValueError, match="output_value"):
        model.encode(DOCUMENTS, is_query=False, output_value="sentence_embedding")
