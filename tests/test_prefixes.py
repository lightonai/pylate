from __future__ import annotations

import json
import os
import tempfile

import pytest

from pylate import models


@pytest.fixture
def base_model_name():
    return "bert-base-uncased"


@pytest.fixture
def model_with_prefix(base_model_name):
    """ColBERT model with default prefixes."""
    return models.ColBERT(model_name_or_path=base_model_name, device="cpu")


@pytest.fixture
def model_custom_prefix(base_model_name):
    """ColBERT model with custom prefix tokens."""
    return models.ColBERT(
        model_name_or_path=base_model_name,
        query_prefix="[QUERY] ",
        document_prefix="[DOC] ",
        device="cpu",
    )


@pytest.fixture
def model_without_prefix(base_model_name):
    """ColBERT model with empty-string prefixes (no prefix insertion)."""
    return models.ColBERT(
        model_name_or_path=base_model_name,
        query_prefix="",
        document_prefix="",
        device="cpu",
    )


class TestPrefixInit:
    """Test that prefix settings are stored correctly at init."""

    def test_default_prefixes(self, base_model_name):
        model = models.ColBERT(model_name_or_path=base_model_name, device="cpu")
        assert model.query_prefix == "[Q] "
        assert model.document_prefix == "[D] "

    def test_empty_string_prefixes(self, model_without_prefix):
        assert model_without_prefix.query_prefix == ""
        assert model_without_prefix.document_prefix == ""

    def test_custom_prefixes(self, model_custom_prefix):
        assert model_custom_prefix.query_prefix == "[QUERY] "
        assert model_custom_prefix.document_prefix == "[DOC] "


class TestTokenizePrefixInsertion:
    """Test that tokenize() inserts/skips prefix tokens based on prefix strings."""

    def test_prefix_inserted_when_set(self, model_with_prefix):
        tokens = model_with_prefix.tokenize(["hello world"], is_query=True)
        input_ids = tokens["input_ids"]
        # Second token (after [CLS]) should be the query prefix
        assert input_ids[0, 1].item() == model_with_prefix.query_prefix_id

    def test_prefix_not_inserted_when_empty(self, model_without_prefix):
        tokens = model_without_prefix.tokenize(["hello world"], is_query=True)
        ref = model_without_prefix._first_module().tokenize(["hello world"])
        # Token at index 1 should be the first content token, not a prefix
        assert tokens["input_ids"][0, 1].item() == ref["input_ids"][0, 1].item()
        assert model_without_prefix.query_prefix_id is None

    def test_document_prefix_inserted_when_set(self, model_with_prefix):
        tokens = model_with_prefix.tokenize(["hello world"], is_query=False)
        input_ids = tokens["input_ids"]
        assert input_ids[0, 1].item() == model_with_prefix.document_prefix_id

    def test_document_prefix_not_inserted_when_empty(self, model_without_prefix):
        tokens = model_without_prefix.tokenize(["hello world"], is_query=False)
        ref = model_without_prefix._first_module().tokenize(["hello world"])
        # Token at index 1 should be the first content token, not a prefix
        assert tokens["input_ids"][0, 1].item() == ref["input_ids"][0, 1].item()
        assert model_without_prefix.document_prefix_id is None

    def test_custom_query_prefix_inserted(self, model_custom_prefix):
        tokens = model_custom_prefix.tokenize(["hello world"], is_query=True)
        input_ids = tokens["input_ids"]
        assert input_ids[0, 1].item() == model_custom_prefix.query_prefix_id

    def test_custom_document_prefix_inserted(self, model_custom_prefix):
        tokens = model_custom_prefix.tokenize(["hello world"], is_query=False)
        input_ids = tokens["input_ids"]
        assert input_ids[0, 1].item() == model_custom_prefix.document_prefix_id

    def test_seq_length_with_custom_prefix(self, model_custom_prefix):
        """Custom prefix should also reduce max_seq_length by 1."""
        model_custom_prefix.tokenize(["hello world"], is_query=True)
        assert (
            model_custom_prefix._first_module().max_seq_length
            == model_custom_prefix.query_length - 1
        )

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
        """With prefix, document tokenization has one extra token (the prefix)."""
        tokens_with = model_with_prefix.tokenize(["hello world"], is_query=False)
        tokens_without = model_without_prefix.tokenize(["hello world"], is_query=False)
        assert (
            tokens_with["input_ids"].shape[1]
            == tokens_without["input_ids"].shape[1] + 1
        )


class TestSaveLoadPrefixes:
    """Test that prefix settings are persisted in config and loaded back."""

    def test_save_load_default_prefixes(self, model_with_prefix):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_with_prefix.save(tmpdir)
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            assert config["query_prefix"] == "[Q] "
            assert config["document_prefix"] == "[D] "

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.query_prefix == "[Q] "
            assert loaded.document_prefix == "[D] "

    def test_save_load_empty_prefixes(self, model_without_prefix):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_without_prefix.save(tmpdir)
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            assert config["query_prefix"] == ""
            assert config["document_prefix"] == ""

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.query_prefix == ""
            assert loaded.document_prefix == ""

    def test_loaded_model_skips_prefix(self, model_without_prefix):
        """A saved model with empty prefixes should skip prefix after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_without_prefix.save(tmpdir)
            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")

            tokens = loaded.tokenize(["hello world"], is_query=True)
            input_ids = tokens["input_ids"]
            # Should not have the query prefix as second token
            assert input_ids[0, 1].item() != loaded.query_prefix_id

    def test_save_load_custom_prefixes(self, model_custom_prefix):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_custom_prefix.save(tmpdir)
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            assert config["query_prefix"] == "[QUERY] "
            assert config["document_prefix"] == "[DOC] "

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.query_prefix == "[QUERY] "
            assert loaded.document_prefix == "[DOC] "

    def test_loaded_custom_prefix_model_inserts_prefix(self, model_custom_prefix):
        """A saved model with custom prefixes should insert them after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_custom_prefix.save(tmpdir)
            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")

            tokens = loaded.tokenize(["hello world"], is_query=True)
            input_ids = tokens["input_ids"]
            assert input_ids[0, 1].item() == loaded.query_prefix_id

    def test_legacy_config_without_prefix_keys_defaults(self, model_with_prefix):
        """Models saved without prefix keys in config should default to [Q]/[D]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_with_prefix.save(tmpdir)
            # Remove the keys from config to simulate a legacy model
            config_path = os.path.join(tmpdir, "config_sentence_transformers.json")
            with open(config_path) as f:
                config = json.load(f)
            del config["query_prefix"]
            del config["document_prefix"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            loaded = models.ColBERT(model_name_or_path=tmpdir, device="cpu")
            assert loaded.query_prefix == "[Q] "
            assert loaded.document_prefix == "[D] "
