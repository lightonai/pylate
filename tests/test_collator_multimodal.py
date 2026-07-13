from __future__ import annotations

from pylate.utils.collator import ColBERTCollator


class TestIsTextColumn:
    """Test ColBERTCollator._is_text_column detection."""

    def test_empty_list(self):
        assert ColBERTCollator._is_text_column([]) is True

    def test_strings(self):
        assert ColBERTCollator._is_text_column(["hello", "world"]) is True

    def test_dicts(self):
        assert ColBERTCollator._is_text_column([{"image": "x"}]) is False

    def test_integers(self):
        assert ColBERTCollator._is_text_column([42, 43]) is False

    def test_none_values(self):
        assert ColBERTCollator._is_text_column([None]) is False


class TestCollatorMultimodalSkipsPrompt:
    """Test that the collator skips prompt prepending for multimodal columns."""

    def test_prompt_not_prepended_to_multimodal(self):
        """Multimodal inputs should not have prompts prepended."""
        calls = []

        def fake_preprocess(texts, is_query=True, pad=False, task=None):
            calls.append({"texts": texts, "is_query": is_query})
            import torch

            n = len(texts)
            return {
                "input_ids": torch.zeros(n, 5, dtype=torch.long),
                "attention_mask": torch.ones(n, 5, dtype=torch.long),
            }

        collator = ColBERTCollator(preprocess_fn=fake_preprocess)
        collator.prompts = {"query": "search: "}

        features = [
            {"query": {"image": "img1.jpg"}, "document": {"image": "img2.jpg"}},
            {"query": {"image": "img3.jpg"}, "document": {"image": "img4.jpg"}},
        ]

        collator(features)

        # The collator should have called preprocess with the raw dicts,
        # NOT with "search: " prepended (which would fail for dicts)
        for call in calls:
            for text in call["texts"]:
                if isinstance(text, str):
                    assert not text.startswith("search: ")

    def test_prompt_prepended_to_text(self):
        """Text inputs should have prompts prepended."""
        calls = []

        def fake_preprocess(texts, is_query=True, pad=False, task=None):
            calls.append({"texts": texts, "is_query": is_query})
            import torch

            n = len(texts)
            return {
                "input_ids": torch.zeros(n, 5, dtype=torch.long),
                "attention_mask": torch.ones(n, 5, dtype=torch.long),
            }

        collator = ColBERTCollator(preprocess_fn=fake_preprocess)
        collator.prompts = {"query": "search: "}

        features = [
            {"query": "what is AI", "document": "AI is cool"},
            {"query": "hello world", "document": "hello back"},
        ]

        collator(features)

        query_call = [c for c in calls if c["is_query"]][0]
        for text in query_call["texts"]:
            assert text.startswith("search: ")
