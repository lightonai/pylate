from __future__ import annotations

import pytest

from pylate import models


@pytest.mark.parametrize(
    "model_name_or_path, revision, query_prefix, document_prefix, max_seq_length, query_length, config",
    [
        (
            "colbert-ir/colbertv2.0",
            "c1e84128e85ef755c096a95bdb06b47793b13acf",
            "[Q]\\s",
            "[D]\\s",
            512,
            32,
            {},
        ),
        (
            "nanoColBERT/ColBERTv1",
            "547fc8b8a87b90a53d1d3f13548aeb5f4caf77c1",
            "[Q]\\s",
            "[D]\\s",
            512,
            32,
            {},
        ),
        (
            "samheym/GerColBERT",
            "d84587b4fd31d66039958d9552f429e7f1a288e7",
            "[unused0]",
            "[unused1]",
            512,
            32,
            {},
        ),
        (
            "lightonai/colbertv2.0",
            "29475aaab88a990378c0d04a63ed819d5c3ba830",
            "[unused0]",
            "[unused1]",
            512,
            32,
            {},
        ),
        (
            "jinaai/jina-colbert-v2",
            "4552c4dc1ffd7d7a635b6a41a1077fe9c9cdd974",
            "[QueryMarker]",
            "[DocumentMarker]",
            8194,
            32,
            {
                "query_prefix": "[QueryMarker]",
                "document_prefix": "[DocumentMarker]",
                "attend_to_expansion_tokens": True,
                "trust_remote_code": True,
            },
        ),
        (
            "antoinelouis/colbert-xm",
            "960de711799d210957d18df59c14c59a439b608a",
            "[unused0]",
            "[unused1]",
            514,
            32,
            {},
        ),
    ],
)
def test_load_model(
    model_name_or_path: str,
    revision: str,
    query_prefix: str,
    document_prefix: str,
    max_seq_length: int,
    query_length: int,
    config: dict,
):
    pylate_model = models.ColBERT(
        model_name_or_path=model_name_or_path, revision=revision, **config
    )

    assert hasattr(pylate_model, "encode")
    assert pylate_model.model_card_data.base_model_revision == revision
    assert pylate_model.model_card_data.library_name == "PyLate"
    assert pylate_model.similarity_fn_name == "MaxSim"
    assert pylate_model.query_prefix == (query_prefix.replace("\\s", " "))
    assert pylate_model.document_prefix == (document_prefix.replace("\\s", " "))
    assert pylate_model.max_seq_length == max_seq_length
    assert pylate_model.query_length == query_length
