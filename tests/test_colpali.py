from __future__ import annotations

import pytest

from pylate import models
from pylate.models.colbert import (
    _COLPALI_TO_BASE_ARCHITECTURE,
)

COLPALI_BASE = "vidore/colqwen2.5-base"
COLPALI_ADAPTER = "vidore/colqwen2.5-v0.2"


class TestDetectColpaliArchitecture:
    def test_detects_colpali_base_model(self):
        """Base ColPali repo has config.json with ColQwen2_5 architecture."""
        arch, config = models.ColBERT._detect_colpali_architecture(COLPALI_BASE, {})
        assert arch is not None
        assert arch in _COLPALI_TO_BASE_ARCHITECTURE
        assert config is not None
        assert config.model_type == "qwen2_5_vl"

    def test_adapter_repo_detected(self):
        """LoRA adapter repos resolve via adapter_config.json -> base model config."""
        arch, config = models.ColBERT._detect_colpali_architecture(COLPALI_ADAPTER, {})
        assert arch is not None
        assert arch in _COLPALI_TO_BASE_ARCHITECTURE
        assert config is not None

    def test_non_colpali_model(self):
        arch, config = models.ColBERT._detect_colpali_architecture(
            "bert-base-uncased", {}
        )
        assert arch is None
        assert config is not None

    def test_invalid_model_path(self):
        arch, config = models.ColBERT._detect_colpali_architecture(
            "this-model-does-not-exist-12345", {}
        )
        assert arch is None
        assert config is None


class TestColpaliBaseModelLoading:
    """Test loading a non-adapter ColPali checkpoint (has config.json)."""

    @pytest.fixture(scope="class")
    def colpali_model(self):
        return models.ColBERT(
            model_name_or_path=COLPALI_BASE,
            device="cpu",
        )

    def test_is_colpali_flag(self, colpali_model):
        assert colpali_model._is_colpali_model is True

    def test_has_two_modules(self, colpali_model):
        module_list = list(colpali_model)
        assert len(module_list) == 2

    def test_dense_projection_dimensions(self, colpali_model):
        module_list = list(colpali_model)
        dense = module_list[1]
        assert dense.in_features > 0
        assert dense.out_features == 128

    def test_encode_text_query(self, colpali_model):
        embeddings = colpali_model.encode(["what is machine learning?"], is_query=True)
        assert len(embeddings) == 1
        assert embeddings[0].ndim == 2
        assert embeddings[0].shape[-1] == 128

    def test_processor_is_base_vlm(self, colpali_model):
        """The processor should be the base VLM processor, not a ColPali one."""
        proc_name = type(colpali_model._first_module().processor).__name__
        assert "Col" not in proc_name

    def test_batched_queries_exclude_padding_tokens(self, colpali_model):
        """Batch-alignment padding must not leak into query embeddings.

        With suffix expansion, only text + expansion tokens are attended; the
        trailing batch padding produces garbage embeddings that would inflate
        MaxSim scores depending on the batch composition.
        """
        queries = [
            "what is machine learning and how is it used in modern applications?",
            "chemistry",
        ]
        features = colpali_model.preprocess(queries, is_query=True)
        attended_lengths = features["attention_mask"].sum(dim=1).tolist()

        embeddings = colpali_model.encode(queries, is_query=True)
        for embedding, attended in zip(embeddings, attended_lengths):
            assert embedding.shape[0] == attended

        # Same queries encoded alone (no batch padding) yield the same counts
        for query, attended in zip(queries, attended_lengths):
            single = colpali_model.encode([query], is_query=True)[0]
            assert single.shape[0] == attended
