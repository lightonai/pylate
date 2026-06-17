"""ViDoRe evaluator for PyLate multi-vector models on visual document retrieval.

Follows the same pattern as NanoBEIREvaluator: loads BEIR-formatted datasets from
the HuggingFace Hub, wraps them in PyLateInformationRetrievalEvaluator instances,
and aggregates NDCG/MRR/Recall across datasets.

Supports three benchmark versions (v1, v2, v3) with per-dataset and per-version
selection.  V2/V3 datasets contain multilingual queries — each language is
evaluated as a separate sub-evaluator (matching MTEB's per-language scoring).
"""

from __future__ import annotations

import gc
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import torch
from datasets import Image as DatasetImage
from PIL import Image
from sentence_transformers.sentence_transformer.evaluation.nano_beir import (
    NanoBEIREvaluator as NanoBEIREvaluatorST,
)
from sentence_transformers.util import is_datasets_available

from .pylate_information_retrieval_evaluator import PyLateInformationRetrievalEvaluator

if TYPE_CHECKING:
    from ..models import ColBERT

logger = logging.getLogger(__name__)


def _decode_lazy_image(value):
    """Decode a datasets lazy-encoded image dict to a PIL Image."""
    if isinstance(value, dict) and "bytes" in value and value["bytes"] is not None:
        return Image.open(BytesIO(value["bytes"]))
    if isinstance(value, dict) and "path" in value and value["path"] is not None:
        return Image.open(value["path"])
    return value


# ── Dataset registry ────────────────────────────────────────────────────────

VIDORE_V1_DATASETS = {
    "arxivqa": "vidore/arxivqa_test_subsampled_beir",
    "docvqa": "vidore/docvqa_test_subsampled_beir",
    "infovqa": "vidore/infovqa_test_subsampled_beir",
    "tabfquad": "vidore/tabfquad_test_subsampled_beir",
    "tatdqa": "vidore/tatdqa_test_beir",
    "shiftproject": "vidore/shiftproject_test_beir",
    "syntheticai": "vidore/syntheticDocQA_artificial_intelligence_test_beir",
    "syntheticenergy": "vidore/syntheticDocQA_energy_test_beir",
    "syntheticgov": "vidore/syntheticDocQA_government_reports_test_beir",
    "synthetichealthcare": "vidore/syntheticDocQA_healthcare_industry_test_beir",
}

VIDORE_V2_DATASETS = {
    "esgreports": "vidore/esg_reports_v2",
    "biomedical": "vidore/biomedical_lectures_v2",
    "economics": "vidore/economics_reports_v2",
}

VIDORE_V3_DATASETS = {
    "finance": "vidore/vidore_v3_finance_en",
    "financefr": "vidore/vidore_v3_finance_fr",
    "hr": "vidore/vidore_v3_hr",
    "industrial": "vidore/vidore_v3_industrial",
    "pharmaceuticals": "vidore/vidore_v3_pharmaceuticals",
    "computerscience": "vidore/vidore_v3_computer_science",
    "energy": "vidore/vidore_v3_energy",
    "physics": "vidore/vidore_v3_physics",
}

VIDORE_VERSION_DATASETS = {
    "v1": VIDORE_V1_DATASETS,
    "v2": VIDORE_V2_DATASETS,
    "v3": VIDORE_V3_DATASETS,
}

ALL_VIDORE_DATASETS = {
    **VIDORE_V1_DATASETS,
    **VIDORE_V2_DATASETS,
    **VIDORE_V3_DATASETS,
}

# CamelCase names — no underscores, required by the parent's metric-key parsing
# (it splits on "_" up to self.name.count("_") to separate dataset prefix from
# the metric suffix like "MaxSim_ndcg@5").
DATASET_NAME_TO_HUMAN_READABLE = {
    "arxivqa": "ArxivQA",
    "docvqa": "DocVQA",
    "infovqa": "InfoVQA",
    "tabfquad": "TabFQuAD",
    "tatdqa": "TATDQA",
    "shiftproject": "ShiftProject",
    "syntheticai": "SyntheticAI",
    "syntheticenergy": "SyntheticEnergy",
    "syntheticgov": "SyntheticGov",
    "synthetichealthcare": "SyntheticHealthcare",
    "esgreports": "ESGReportsV2",
    "biomedical": "BiomedicalV2",
    "economics": "EconomicsV2",
    "finance": "FinanceV3",
    "financefr": "FinanceFrV3",
    "hr": "HRV3",
    "industrial": "IndustrialV3",
    "pharmaceuticals": "PharmaceuticalsV3",
    "computerscience": "ComputerScienceV3",
    "energy": "EnergyV3",
    "physics": "PhysicsV3",
}

VIDORE_V2_LANGUAGES = ["english", "french", "spanish", "german"]
VIDORE_V3_LANGUAGES = [
    "english",
    "french",
    "spanish",
    "german",
    "italian",
    "portuguese",
]

LANGUAGE_SHORT = {
    "english": "En",
    "french": "Fr",
    "spanish": "Es",
    "german": "De",
    "italian": "It",
    "portuguese": "Pt",
}

DatasetNameType = Literal[
    "arxivqa",
    "docvqa",
    "infovqa",
    "tabfquad",
    "tatdqa",
    "shiftproject",
    "syntheticai",
    "syntheticenergy",
    "syntheticgov",
    "synthetichealthcare",
    "esgreports",
    "biomedical",
    "economics",
    "finance",
    "financefr",
    "hr",
    "industrial",
    "pharmaceuticals",
    "computerscience",
    "energy",
    "physics",
]

VersionType = Literal["v1", "v2", "v3"]


# ── Multimodal IR evaluator ────────────────────────────────────────────────


class ViDoREInformationRetrievalEvaluator(PyLateInformationRetrievalEvaluator):
    """IR evaluator that decodes lazy-encoded corpus images before encoding.

    ViDoRe corpora store images with ``datasets.Image(decode=False)`` to avoid
    materializing every PIL image at init time.  This subclass decodes the
    ``{"bytes": ..., "path": ...}`` dicts to PIL Images per-chunk right before
    the model encodes them.
    """

    @staticmethod
    def _decode_corpus_chunk(entries: list) -> list:
        return [
            {**entry, "image": _decode_lazy_image(entry["image"])}
            if isinstance(entry, dict) and "image" in entry
            else entry
            for entry in entries
        ]

    def _get_corpus_chunk(self, start: int, end: int) -> list:
        return self._decode_corpus_chunk(self.corpus[start:end])


# ── Evaluator ───────────────────────────────────────────────────────────────


class ViDoREvaluator(NanoBEIREvaluatorST):
    """Evaluate a PyLate ColBERT model on ViDoRe visual document retrieval.

    Mirrors :class:`~pylate.evaluation.NanoBEIREvaluator` but loads multimodal
    (image) corpora from the ``vidore/*`` datasets on the HuggingFace Hub.

    V2/V3 datasets contain multilingual queries.  Each language is evaluated
    as a separate sub-evaluator (matching MTEB).  When ``language`` is set,
    only that language is evaluated; when ``None`` (default), every available
    language for each dataset gets its own sub-evaluator.

    Parameters
    ----------
    dataset_names : list[str] | None
        Specific dataset short names to evaluate.  Overrides *versions* when
        provided.  Valid names are listed in :data:`ALL_VIDORE_DATASETS`.
    versions : list[str] | None
        Benchmark versions to include — any combination of ``"v1"``,
        ``"v2"``, ``"v3"``.  Defaults to ``["v1"]``.
    language : str | None
        Evaluate only this language for v2/v3 multilingual datasets.
        E.g. ``"english"``, ``"french"``, ``"german"``, ``"spanish"``,
        ``"italian"``, ``"portuguese"``.
        ``None`` (default) evaluates every language separately (one
        sub-evaluator per dataset-language pair, matching MTEB).
        V1 datasets are monolingual and always included as-is.
    document_prompt : str | None
        Text appended alongside each corpus image
        (e.g. ``"Describe the image."``).  Defaults to ``None`` (raw images,
        matching MTEB's processing).
    corpus_chunk_size : int
        Number of corpus documents encoded per forward-pass chunk.
        Keep small (32-64) for image corpora to avoid OOM.
    ndcg_at_k : list[int] | None
        NDCG cut-offs.  Defaults to ``[5]`` for v1/v2, ``[5, 10]`` when v3
        datasets are included (MTEB uses NDCG@10 as main score for v3).
    batch_size : int
        Encoding batch size.
    **kwargs
        Forwarded to the parent (``mrr_at_k``, ``accuracy_at_k``, ``map_at_k``,
        ``show_progress_bar``, ``write_csv``, ``aggregate_fn``, etc.).

    Examples
    --------
    from pylate import models, evaluation

    model = models.ColBERT(model_name_or_path="my-vidore-model", device="cuda")

    All v1 datasets (default):

    evaluator = evaluation.ViDoREvaluator()

    Specific versions:

    evaluator = evaluation.ViDoREvaluator(versions=["v1", "v2"])

    Cherry-pick datasets across versions:

    evaluator = evaluation.ViDoREvaluator(dataset_names=["arxivqa", "infovqa", "finance"])

    Only French queries on v3:

    evaluator = evaluation.ViDoREvaluator(versions=["v3"], language="french")

    Use in a training loop:

    from sentence_transformers import SentenceTransformerTrainer
    trainer = SentenceTransformerTrainer(..., evaluator=evaluator)
    """

    information_retrieval_class = ViDoREInformationRetrievalEvaluator

    def __init__(
        self,
        dataset_names: list[DatasetNameType | str] | None = None,
        versions: list[VersionType] | None = None,
        language: str | None = None,
        document_prompt: str | None = None,
        corpus_chunk_size: int = 32,
        ndcg_at_k: list[int] | None = None,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        # Must be set before super().__init__ calls _load_dataset
        self.document_prompt = document_prompt
        self._corpus_chunk_size = corpus_chunk_size

        if dataset_names is None:
            if versions is None:
                versions = ["v1"]
            dataset_names = []
            for v in versions:
                if v not in VIDORE_VERSION_DATASETS:
                    raise ValueError(
                        f"Unknown version {v!r}. "
                        f"Valid versions: {list(VIDORE_VERSION_DATASETS)}"
                    )
                dataset_names.extend(VIDORE_VERSION_DATASETS[v].keys())

        resolved_lang = language.lower() if language else None

        # Expand multilingual datasets (v2/v3) into per-language entries.
        # Format: "datasetname:language" for multilingual, plain name for v1.
        expanded_names: list[str] = []
        for dn in dataset_names:
            dn_lower = dn.lower()
            if dn_lower in VIDORE_V2_DATASETS:
                langs = [resolved_lang] if resolved_lang else VIDORE_V2_LANGUAGES
                for lang in langs:
                    expanded_names.append(f"{dn_lower}:{lang}")
            elif dn_lower in VIDORE_V3_DATASETS:
                langs = [resolved_lang] if resolved_lang else VIDORE_V3_LANGUAGES
                for lang in langs:
                    expanded_names.append(f"{dn_lower}:{lang}")
            else:
                expanded_names.append(dn_lower)
        dataset_names = expanded_names

        has_v3 = any(dn.split(":")[0] in VIDORE_V3_DATASETS for dn in dataset_names)
        if ndcg_at_k is None:
            ndcg_at_k = [5, 10] if has_v3 else [5]
        elif has_v3 and 10 not in ndcg_at_k:
            ndcg_at_k = list(ndcg_at_k) + [10]

        super().__init__(
            dataset_names=dataset_names,
            ndcg_at_k=ndcg_at_k,
            batch_size=batch_size,
            **kwargs,
        )

        aggregate_key = kwargs.get("aggregate_key", "mean")
        self.name = f"ViDoRE_{aggregate_key}"
        if self.truncate_dim:
            self.name += f"_{self.truncate_dim}"
        self.csv_file = f"ViDoRE_evaluation_{aggregate_key}_results.csv"

    # ── Overrides ───────────────────────────────────────────────────────────

    def _validate_dataset_names(self) -> None:
        if not self.dataset_names:
            raise ValueError(
                "dataset_names cannot be empty. "
                "Pass None to evaluate on the default version (v1)."
            )
        missing = []
        for n in self.dataset_names:
            base = n.split(":")[0].lower()
            if base not in ALL_VIDORE_DATASETS:
                missing.append(n)
        if missing:
            raise ValueError(
                f"Unknown ViDoRe dataset(s): {missing}. "
                f"Valid names: {sorted(ALL_VIDORE_DATASETS)}"
            )

    def _get_human_readable_name(self, dataset_name: str) -> str:
        if ":" in dataset_name:
            base, lang = dataset_name.rsplit(":", 1)
            base_hr = DATASET_NAME_TO_HUMAN_READABLE[base.lower()]
            lang_short = LANGUAGE_SHORT.get(lang, lang.capitalize())
            name = f"{base_hr}{lang_short}"
        else:
            name = DATASET_NAME_TO_HUMAN_READABLE[dataset_name.lower()]
        if self.truncate_dim is not None:
            name += f"_{self.truncate_dim}"
        return name

    def _load_dataset(
        self, dataset_name: str, **ir_evaluator_kwargs
    ) -> PyLateInformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Install it with: pip install datasets"
            )
        from datasets import load_dataset

        if ":" in dataset_name:
            base_name, lang = dataset_name.rsplit(":", 1)
        else:
            base_name, lang = dataset_name, None

        dataset_path = ALL_VIDORE_DATASETS[base_name.lower()]

        queries_ds = load_dataset(dataset_path, "queries", split="test")
        corpus_ds = load_dataset(dataset_path, "corpus", split="test")
        qrels_ds = load_dataset(dataset_path, "qrels", split="test")

        # v1/v2 use hyphenated keys, v3 uses underscores
        qid_col = "query-id" if "query-id" in queries_ds.column_names else "query_id"
        cid_col = "corpus-id" if "corpus-id" in corpus_ds.column_names else "corpus_id"
        qrel_qid = "query-id" if "query-id" in qrels_ds.column_names else "query_id"
        qrel_cid = "corpus-id" if "corpus-id" in qrels_ds.column_names else "corpus_id"

        if lang and "language" in queries_ds.column_names:
            queries_ds = queries_ds.filter(lambda r: r["language"] == lang)

        # Keep corpus images as encoded bytes (decode=False) to avoid
        # materializing every PIL image at init time.  Images are decoded
        # lazily in PyLateInformationRetrievalEvaluator.compute_all_metrics
        # right before each chunk is encoded.
        image_col = "image"
        if image_col in corpus_ds.column_names:
            feat = corpus_ds.features.get(image_col)
            if isinstance(feat, DatasetImage) and feat.decode:
                corpus_ds = corpus_ds.cast_column(image_col, DatasetImage(decode=False))

        queries = {str(r[qid_col]): r["query"] for r in queries_ds}

        if self.document_prompt:
            corpus = {
                str(r[cid_col]): {"image": r["image"], "text": self.document_prompt}
                for r in corpus_ds
            }
        else:
            corpus = {str(r[cid_col]): {"image": r["image"]} for r in corpus_ds}

        relevant_docs: dict[str, set[str]] = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r[qrel_qid])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r[qrel_cid]))

        human_readable_name = self._get_human_readable_name(dataset_name)

        ir_evaluator_kwargs["corpus_chunk_size"] = self._corpus_chunk_size

        return self.information_retrieval_class(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def __call__(
        self,
        model: ColBERT,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        results = super().__call__(model, output_path, epoch, steps, *args, **kwargs)

        # Free GPU memory accumulated during evaluation (many sub-evaluators
        # each encode corpus images, fragmenting the CUDA allocator).  Without
        # this, training can hang on the first all_gather after eval in DDP.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_underscores = self.name.count("_")

        # Compute per-dataset macro averages over languages (v2/v3 only)
        dataset_lang_groups: dict[str, list[str]] = {}
        for dataset_name in self.dataset_names:
            if ":" in dataset_name:
                base = dataset_name.rsplit(":", 1)[0].lower()
                dataset_lang_groups.setdefault(base, []).append(
                    self._get_human_readable_name(dataset_name)
                )

        for base, hr_names in dataset_lang_groups.items():
            base_hr = DATASET_NAME_TO_HUMAN_READABLE[base]
            if self.truncate_dim is not None:
                base_hr += f"_{self.truncate_dim}"

            per_metric: dict[str, list[float]] = {}
            for hr_name in hr_names:
                prefix = hr_name + "_"
                for key, value in results.items():
                    if key.startswith(prefix):
                        metric = key.split("_", maxsplit=num_underscores)[-1]
                        per_metric.setdefault(metric, []).append(value)

            for metric, values in per_metric.items():
                results[f"{base_hr}_{metric}"] = sum(values) / len(values)

            ndcg_k = 10 if base in VIDORE_V3_DATASETS else 5
            for score_name in self.score_function_names:
                avg_key = f"{base_hr}_{score_name}_ndcg@{ndcg_k}"
                if avg_key in results:
                    logger.warning(
                        f"{base_hr} macro NDCG@{ndcg_k}: {results[avg_key]:.4f}"
                    )

        # Group evaluated datasets by version
        version_groups: dict[str, list[str]] = {}
        for dataset_name in self.dataset_names:
            base = dataset_name.split(":")[0].lower()
            for version, datasets in VIDORE_VERSION_DATASETS.items():
                if base in datasets:
                    version_groups.setdefault(version, []).append(
                        self._get_human_readable_name(dataset_name)
                    )
                    break

        # Compute per-version macro averages
        for version, hr_names in version_groups.items():
            per_metric: dict[str, list[float]] = {}
            for hr_name in hr_names:
                prefix = hr_name + "_"
                for key, value in results.items():
                    if key.startswith(prefix):
                        metric = key.split("_", maxsplit=num_underscores)[-1]
                        per_metric.setdefault(metric, []).append(value)

            for metric, values in per_metric.items():
                results[f"ViDoRE_{version}_{metric}"] = sum(values) / len(values)

            ndcg_k = 10 if version == "v3" else 5
            for score_name in self.score_function_names:
                ver_key = f"ViDoRE_{version}_{score_name}_ndcg@{ndcg_k}"
                if ver_key in results:
                    logger.warning(
                        f"ViDoRE {version} macro NDCG@{ndcg_k}: {results[ver_key]:.4f}"
                    )

        return results
