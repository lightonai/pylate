from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from platform import python_version
from typing import TYPE_CHECKING, Any, Literal

import torch
import transformers
from huggingface_hub import ModelCard
from sentence_transformers import SentenceTransformerModelCardData
from sentence_transformers import __version__ as sentence_transformers_version
from sentence_transformers.util import (
    is_accelerate_available,
    is_datasets_available,
)
from torch import nn
from transformers.integrations import CodeCarbonCallback

from ..__version__ import __version__ as pylate_version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from sentence_transformers.trainer import SentenceTransformerTrainer


IGNORED_FIELDS = ["model", "trainer", "eval_results_dict"]


def get_versions() -> dict[str, Any]:
    versions = {
        "python": python_version(),
        "sentence_transformers": sentence_transformers_version,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "pylate": pylate_version,
    }
    if is_accelerate_available():
        from accelerate import __version__ as accelerate_version

        versions["accelerate"] = accelerate_version
    if is_datasets_available():
        from datasets import __version__ as datasets_version

        versions["datasets"] = datasets_version
    from tokenizers import __version__ as tokenizers_version

    versions["tokenizers"] = tokenizers_version
    return versions


@dataclass
class PylateModelCardData(SentenceTransformerModelCardData):
    """
    A dataclass for storing data used in the model card.

    Parameters
    ----------
    language
        The model language, either a string or a list of strings, e.g., "en" or ["en", "de", "nl"].
    license
        The license of the model, e.g., "apache-2.0", "mit", or "cc-by-nc-sa-4.0".
    model_name
        The pretty name of the model, e.g., "SentenceTransformer based on microsoft/mpnet-base".
    model_id
        The model ID for pushing the model to the Hub, e.g., "tomaarsen/sbert-mpnet-base-allnli".
    train_datasets
        A list of dictionaries containing names and/or Hugging Face dataset IDs for training datasets,
        e.g., [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}].
    eval_datasets
        A list of dictionaries containing names and/or Hugging Face dataset IDs for evaluation datasets,
        e.g., [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}].
    task_name
        The human-readable task the model is trained on, e.g., "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".
    tags
        A list of tags for the model, e.g., ["sentence-transformers", "sentence-similarity", "feature-extraction"].
    """

    # Potentially provided by the user
    language: str | list[str] | None = field(default_factory=list)
    license: str | None = None
    model_name: str | None = None
    model_id: str | None = None
    train_datasets: list[dict[str, str]] = field(default_factory=list)
    eval_datasets: list[dict[str, str]] = field(default_factory=list)
    task_name: str = "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more"
    tags: list[str] | None = field(
        default_factory=lambda: [
            "ColBERT",
            "PyLate",
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
        ]
    )
    generate_widget_examples: Literal["deprecated"] = "deprecated"

    # Automatically filled by `ModelCardCallback` and the Trainer directly
    base_model: str | None = field(default=None, init=False)
    base_model_revision: str | None = field(default=None, init=False)
    non_default_hyperparameters: dict[str, Any] = field(
        default_factory=dict, init=False
    )
    all_hyperparameters: dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: dict[SentenceEvaluator, dict[str, Any]] | None = field(
        default_factory=dict, init=False
    )
    training_logs: list[dict[str, float]] = field(default_factory=list, init=False)
    widget: list[dict[str, str]] = field(default_factory=list, init=False)
    predict_example: str | None = field(default=None, init=False)
    label_example_list: list[dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: CodeCarbonCallback | None = field(default=None, init=False)
    citations: dict[str, str] = field(default_factory=dict, init=False)
    best_model_step: int | None = field(default=None, init=False)
    trainer: SentenceTransformerTrainer | None = field(
        default=None, init=False, repr=False
    )
    datasets: list[str] = field(default_factory=list, init=False, repr=False)

    # Utility fields
    first_save: bool = field(default=True, init=False)
    widget_step: int = field(default=-1, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default="sentence-similarity", init=False)
    library_name: str = field(default="PyLate", init=False)
    version: dict[str, str] = field(default_factory=get_versions, init=False)

    # Passed via `register_model` only
    model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    def set_losses(self, losses: list[nn.Module]) -> None:
        citations = {
            "Sentence Transformers": """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}""",
            "PyLate": """
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}""",
        }
        for loss in losses:
            try:
                citations[loss.__class__.__name__] = loss.citation
            except Exception:
                pass
        inverted_citations = defaultdict(list)
        for loss, citation in citations.items():
            inverted_citations[citation].append(loss)

        def join_list(losses: list[str]) -> str:
            if len(losses) > 1:
                return ", ".join(losses[:-1]) + " and " + losses[-1]
            return losses[0]

        self.citations = {
            join_list(losses): citation
            for citation, losses in inverted_citations.items()
        }
        self.add_tags(
            [
                f"loss:{loss}"
                for loss in {loss.__class__.__name__: loss for loss in losses}
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        # Try to set the base model
        if self.first_save and not self.base_model:
            try:
                self.try_to_set_base_model()
            except Exception:
                pass

        # Set the model name
        if not self.model_name:
            if self.base_model:
                self.model_name = f"PyLate model based on {self.base_model}"
            else:
                self.model_name = "PyLate"

        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Compute required formats from the (usually post-training) evaluation data
        if self.eval_results_dict:
            try:
                super_dict.update(self.format_eval_metrics())
            except Exception as exc:
                logger.warning(f"Error while formatting evaluation metrics: {exc}")
                raise exc

        # Compute required formats for the during-training evaluation data
        if self.training_logs:
            try:
                super_dict.update(self.format_training_logs())
            except Exception as exc:
                logger.warning(f"Error while formatting training logs: {exc}")

        super_dict["hide_eval_lines"] = len(self.training_logs) > 100

        # Try to add the code carbon callback data
        if (
            self.code_carbon_callback
            and self.code_carbon_callback.tracker
            and self.code_carbon_callback.tracker._start_time is not None
        ):
            super_dict.update(self.get_codecarbon_data())

        # Add some additional metadata stored in the model itself
        super_dict["document_length"] = self.model.document_length
        super_dict["query_length"] = self.model.query_length
        super_dict["output_dimensionality"] = (
            self.model.get_sentence_embedding_dimension()
        )
        super_dict["model_string"] = str(self.model)
        if self.model.similarity_fn_name:
            super_dict["similarity_fn_name"] = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(
                self.model.similarity_fn_name,
                self.model.similarity_fn_name.replace("_", " ").title(),
            )
        else:
            super_dict["similarity_fn_name"] = "Cosine Similarity"

        self.first_save = False

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return super_dict

    # For now, set_widget_examples is not compatible with our transform/map operations, so we make it a no-op until it is fixed
    def set_widget_examples(self, dataset) -> None:
        pass


def generate_model_card(model: SentenceTransformer) -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    model_card = ModelCard.from_template(
        card_data=model.model_card_data, template_path=template_path, hf_emoji="🐕"
    )
    return model_card.content
