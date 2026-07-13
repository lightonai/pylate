from __future__ import annotations

import pytest

from pylate import models


@pytest.fixture
def model():
    return models.ColBERT(
        model_name_or_path="bert-base-uncased",
        device="cpu",
    )


class TestAppendExpansionTokens:
    def test_appends_correct_number_of_tokens(self, model):
        features = model._first_module().preprocess(["hello world"])
        original_len = features["input_ids"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=10)
        assert result["input_ids"].shape[1] == original_len + 10

    def test_appended_tokens_are_pad_id(self, model):
        features = model._first_module().preprocess(["hello world"])
        original_len = features["input_ids"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=5)
        pad_id = model.tokenizer.pad_token_id
        appended = result["input_ids"][0, original_len:]
        assert (appended == pad_id).all()

    def test_attention_mask_extended(self, model):
        features = model._first_module().preprocess(["hello world"])
        original_len = features["attention_mask"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=7)
        assert result["attention_mask"].shape[1] == original_len + 7
        # Appended attention mask entries should be 1
        assert (result["attention_mask"][0, original_len:] == 1).all()

    def test_token_type_ids_extended_with_zeros(self, model):
        features = model._first_module().preprocess(["hello world"])
        assert "token_type_ids" in features
        original_len = features["token_type_ids"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=4)
        assert result["token_type_ids"].shape[1] == original_len + 4
        assert (result["token_type_ids"][0, original_len:] == 0).all()

    def test_n_tokens_zero(self, model):
        features = model._first_module().preprocess(["hello world"])
        original_len = features["input_ids"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=0)
        assert result["input_ids"].shape[1] == original_len

    def test_batch_of_multiple_inputs(self, model):
        features = model._first_module().preprocess(["hello", "world foo bar"])
        batch_size = features["input_ids"].shape[0]
        original_len = features["input_ids"].shape[1]
        result = model._append_expansion_tokens(features, n_tokens=3)
        assert result["input_ids"].shape == (batch_size, original_len + 3)

    def test_suffix_expansion_via_preprocess(self, model):
        """When _is_colpali_model is True and do_query_expansion is True,
        preprocess should use suffix expansion (append tokens) instead of
        pad-to-fixed-length expansion."""
        model._is_colpali_model = True
        model.do_query_expansion = True
        tokens_a = model.preprocess(["short"], is_query=True)
        tokens_b = model.preprocess(["a much longer query text here"], is_query=True)
        len_a = tokens_a["input_ids"].shape[1]
        len_b = tokens_b["input_ids"].shape[1]
        # Suffix expansion: different query lengths should produce different output lengths
        # (unlike pad-to-fixed-length which would make them equal)
        assert len_a != len_b
