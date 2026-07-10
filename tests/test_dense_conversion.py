from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense as STDense
from sentence_transformers.models import Pooling

from pylate import models
from pylate.models.Dense import Dense


def test_st_checkpoint_with_sentence_level_dense_converts(tmp_path):
    """ST checkpoints with a Dense after Pooling (e.g. sentence-t5, LaBSE)
    must convert to a token-level projection instead of crashing with
    KeyError('sentence_embedding') once Pooling is filtered out."""
    base = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    transformer = base[0]
    hidden = transformer.get_embedding_dimension()
    st_dense = STDense(
        in_features=hidden,
        out_features=64,
        bias=False,
        activation_function=nn.Identity(),
    )
    st_model = SentenceTransformer(
        modules=[transformer, Pooling(hidden), st_dense], device="cpu"
    )
    path = str(tmp_path / "st_dense_model")
    st_model.save(path)

    colbert = models.ColBERT(path, device="cpu")

    assert [type(m).__name__ for m in colbert] == ["Transformer", "Dense"]
    assert colbert[1].module_input_name == "token_embeddings"
    assert colbert[1].module_output_name == "token_embeddings"
    assert torch.equal(colbert[1].linear.weight, st_dense.linear.weight)

    embeddings = colbert.encode(["hello world"], is_query=True)
    assert embeddings[0].ndim == 2
    assert embeddings[0].shape[-1] == 64


def test_from_sentence_transformers_preserves_token_level_dense():
    """A Dense already operating on token embeddings converts unchanged."""
    st_dense = STDense(
        in_features=32,
        out_features=16,
        bias=False,
        activation_function=nn.Identity(),
        module_input_name="token_embeddings",
        module_output_name="token_embeddings",
    )
    converted = Dense.from_sentence_transformers(st_dense)
    assert converted.module_input_name == "token_embeddings"
    assert torch.equal(converted.linear.weight, st_dense.linear.weight)

    features = {"token_embeddings": torch.randn(2, 5, 32)}
    out = converted(features)
    assert out["token_embeddings"].shape == (2, 5, 16)
