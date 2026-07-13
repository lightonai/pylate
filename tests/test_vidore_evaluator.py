from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pylate.evaluation import ViDoREvaluator


@pytest.fixture
def no_dataset_loading(monkeypatch):
    """Stub _load_dataset so init-time validation can be tested offline."""
    monkeypatch.setattr(
        ViDoREvaluator,
        "_load_dataset",
        lambda self, name, **kwargs: MagicMock(name=name),
    )


def test_unknown_language_raises_at_init(no_dataset_loading):
    with pytest.raises(ValueError, match="Unknown language 'klingon'"):
        ViDoREvaluator(versions=["v3"], language="klingon")


def test_language_unavailable_for_all_selected_raises(no_dataset_loading):
    # italian exists in v3 only; a pure-v2 selection has nothing to evaluate
    with pytest.raises(ValueError, match="No datasets left"):
        ViDoREvaluator(dataset_names=["esgreports"], language="italian")


def test_language_skips_unavailable_datasets(no_dataset_loading, caplog):
    # Mixed v2+v3 selection: the v2 dataset is skipped with a warning, the
    # v3 dataset is evaluated in the requested language.
    with caplog.at_level("WARNING", logger="pylate.evaluation.vidore_evaluator"):
        evaluator = ViDoREvaluator(
            dataset_names=["esgreports", "hr"], language="italian"
        )
    assert evaluator.dataset_names == ["hr:italian"]
    assert any("Skipping esgreports" in r.message for r in caplog.records)


def test_language_expansion_without_language(no_dataset_loading):
    evaluator = ViDoREvaluator(dataset_names=["esgreports"])
    assert evaluator.dataset_names == [
        "esgreports:english",
        "esgreports:french",
        "esgreports:spanish",
        "esgreports:german",
    ]


def test_language_case_insensitive(no_dataset_loading):
    evaluator = ViDoREvaluator(dataset_names=["esgreports"], language="French")
    assert evaluator.dataset_names == ["esgreports:french"]


@pytest.fixture
def fake_hub_datasets(monkeypatch):
    """Replace hub loading with a tiny in-memory v1-style dataset."""
    import datasets

    def fake_load_dataset(path, name, split):
        if name == "queries":
            return datasets.Dataset.from_dict(
                {"query-id": [1, 2], "query": ["q one", "q two"]}
            )
        if name == "corpus":
            return datasets.Dataset.from_dict(
                {"corpus-id": [10, 11], "image": ["img-a", "img-b"]}
            )
        if name == "qrels":
            return datasets.Dataset.from_dict(
                {"query-id": [1, 2], "corpus-id": [10, 11], "score": [1, 1]}
            )
        raise AssertionError(name)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)


def test_query_prompts_forwarded_to_sub_evaluators(fake_hub_datasets):
    evaluator = ViDoREvaluator(
        dataset_names=["arxivqa"], query_prompts="Find a relevant screenshot: "
    )
    assert evaluator.evaluators[0].query_prompt == "Find a relevant screenshot: "


def test_corpus_prompts_injected_as_image_text(fake_hub_datasets):
    evaluator = ViDoREvaluator(
        dataset_names=["arxivqa"], corpus_prompts="Describe the image."
    )
    corpus = evaluator.evaluators[0].corpus
    assert all(entry["text"] == "Describe the image." for entry in corpus)
    # Not forwarded as an encode-time string prefix (meaningless for images)
    assert not getattr(evaluator.evaluators[0], "corpus_prompt", None)


def test_no_corpus_prompts_keeps_raw_images(fake_hub_datasets):
    evaluator = ViDoREvaluator(dataset_names=["arxivqa"])
    corpus = evaluator.evaluators[0].corpus
    assert all("text" not in entry for entry in corpus)
