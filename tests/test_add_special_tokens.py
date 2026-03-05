from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
import torch

from pylate import models


@pytest.fixture
def base_model_name():
    return "bert-base-uncased"


@pytest.fixture
def model_with_prefix(base_model_name):
    """ColBERT model with default add_special_tokens=True."""
    return models.ColBERT(model_name_or_path=base_model_name, device="cpu")


@pytest.fixture
def model_without_prefix(base_model_name):
    """ColBERT model with add_special_tokens=False."""
    return models.ColBERT(
        model_name_or_path=base_model_name, add_special_tokens=False, device="cpu"
    )


class TestAddSpecialTokensInit:
    """Test that add_special_tokens is stored correctly at init."""

    def test_default_is_true(self, base_model_name):
        model = models.ColBERT(model_name_or_path=base_model_name, device="cpu")
        assert model.add_special_tokens is True

    def test_explicit_false(self, model_without_prefix):
        assert model_without_prefix.add_special_tokens is False

    def test_explicit_true(self, base_model_name):
        model = models.ColBERT(
            model_name_or_path=base_model_name,
            add_special_tokens=True,
            device="cpu",
        )
        assert model.add_special_tokens is True


class TestTokenizePrefixInsertion:
    """Test that tokenize() inserts/skips prefix tokens based on add_special_tokens."""

    def test_prefix_inserted_when_true(self, model_with_prefix):
        tokens = model_with_prefix.tokenize(["hello world"], is_query=True)
        input_ids = tokens["input_ids"]
        # Second token (after [CLS]) should be the query prefix
        assert input_ids[0, 1].item() == model_with_prefix.query_prefix_id

    def test_prefix_not_inserted_when_false(self, model_without_prefix):
        tokens = model_without_prefix.tokenize(["hello world"], is_query=True)
        input_ids = tokens["input_ids"]
        # Second token (after [CLS]) should NOT be the query prefix
        assert input_ids[0, 1].item() != model_without_prefix.query_prefix_id

    def test_document_prefix_inserted_when_true(self, model_with_prefix):
        tokens = model_with_prefix.tokenize(["hello world"], is_query=False)
        input_ids = tokens["input_ids"]
        assert input_ids[0, 1].item() == model_with_prefix.document_prefix_id

    def test_document_prefix_not_inserted_when_false(self, model_without_prefix):
        tokens = model_without_prefix.tokenize(["hello world"], is_query=False)
        input_ids = tokens["input_ids"]
        assert input_ids[0, 1].item() != model_without_prefix.document_prefix_id

    def test_seq_length_with_prefix(self, model_with_prefix):
        """With prefix, max_seq_length should be max_length - 1."""
        model_with_prefix.tokenize(["hello world"], is_query=True)
        assert (
            model_with_prefix._first_module().max_seq_length
            == model_with_prefix.query_length - 1
        )

    def test_seq_length_without_prefix(self, model_without_prefix):
        """Without prefix, max_seq_length should be max_length (no -1)."""
        model_without_prefix.tokenize(["hello world"], is_query=True)
        assert (
            model_without_prefix._first_module().max_seq_length
            == model_without_prefix.query_length
        )

    def test_output_length_difference(self, model_with_prefix, model_without_prefix):
        """With prefix, output has one extra token (the prefix) compared to without."""
        tokens_with = model_with_prefix.tokenize(["hello world"], is_query=True)
        tokens_without = model_without_prefix.tokenize(["hello world"], is_query=True)
        len_with = tokens_with["input_ids"].shape[1]
        len_without = tokens_without["input_ids"].shape[1]
        # Both should pad to query_length, but with-prefix has the prefix token occupying one slot
        assert len_with == model_with_prefix.query_length
        assert len_without == model_without_prefix.query_length


class TestSaveLoadAddSpecialTokens:
    """Test that add_special_tokens is persisted in config and loaded back."""

    def test_save_load_true(self, model_with_prefix):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_with_prefix.save(tmpdir)
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            assert config["add_special_tokens"] is True

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.add_special_tokens is True

    def test_save_load_false(self, model_without_prefix):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_without_prefix.save(tmpdir)
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            assert config["add_special_tokens"] is False

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.add_special_tokens is False

    def test_loaded_model_skips_prefix(self, model_without_prefix):
        """A saved model with add_special_tokens=False should skip prefix after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_without_prefix.save(tmpdir)
            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")

            tokens = loaded.tokenize(["hello world"], is_query=True)
            input_ids = tokens["input_ids"]
            # Should not have the query prefix as second token
            assert input_ids[0, 1].item() != loaded.query_prefix_id

    def test_legacy_config_without_key_defaults_true(self, model_with_prefix):
        """Models saved without add_special_tokens in config should default to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_with_prefix.save(tmpdir)
            # Remove the key from config to simulate a legacy model
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            del config["add_special_tokens"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            # Default from __init__ signature is True
            assert loaded.add_special_tokens is True
