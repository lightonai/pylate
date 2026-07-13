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
import os
from contextlib import nullcontext
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from datasets import Image as DatasetImage
from PIL import Image
from sentence_transformers.sentence_transformer.evaluation.nano_beir import (
    NanoBEIREvaluator as NanoBEIREvaluatorST,
)
from sentence_transformers.util import is_datasets_available
from tqdm import tqdm, trange

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

    .. note::
        Relevance judgments are binarized (score > 0), as the underlying
        sentence-transformers evaluator only supports binary relevance.
        V1/v2 qrels carry a single grade so this is lossless, but v3 qrels
        are graded (1 and 2): v3 NDCG values are therefore not directly
        comparable to the MTEB leaderboard, which computes graded-gain NDCG
        via pytrec_eval. Rankings between models remain meaningful.

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
    corpus_prompts : str | dict[str, str], optional
        Text rendered alongside each corpus image
        (e.g. ``"Describe the image."``) — injected as the image-side
        ``"text"``, not a string prefix.  Defaults to ``None`` (raw images,
        matching MTEB's processing).  A string applies to all datasets; a
        dict must be keyed by the expanded ``"dataset:language"`` names.
        Make sure this matches how the model was trained.
    query_prompts : str | dict[str, str], optional
        Prompt(s) prepended to queries at encode time (e.g. MTEB v3 uses
        ``"Find a screenshot that is relevant to the user's question."``).
        A string applies to all datasets; a dict must be keyed by the
        expanded ``"dataset:language"`` names.  Make sure this matches how
        the model was trained.
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
        corpus_chunk_size: int = 32,
        ndcg_at_k: list[int] | None = None,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        # Must be set before super().__init__ calls _load_dataset
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
        if resolved_lang is not None and resolved_lang not in LANGUAGE_SHORT:
            raise ValueError(
                f"Unknown language {language!r}. "
                f"Supported languages: {list(LANGUAGE_SHORT)}"
            )

        # Expand multilingual datasets (v2/v3) into per-language entries.
        # Format: "datasetname:language" for multilingual, plain name for v1.
        # When a language is requested, datasets that do not cover it are
        # skipped with a warning (e.g. italian only exists in v3), so that
        # mixed-version selections still evaluate the language where available.
        expanded_names: list[str] = []
        for dn in dataset_names:
            dn_lower = dn.lower()
            if dn_lower in VIDORE_V2_DATASETS:
                available_langs = VIDORE_V2_LANGUAGES
            elif dn_lower in VIDORE_V3_DATASETS:
                available_langs = VIDORE_V3_LANGUAGES
            else:
                expanded_names.append(dn_lower)
                continue
            if resolved_lang is not None:
                if resolved_lang not in available_langs:
                    logger.warning(
                        f"Skipping {dn_lower}: language {resolved_lang!r} is not "
                        f"available for this dataset (available: {available_langs})."
                    )
                    continue
                langs = [resolved_lang]
            else:
                langs = available_langs
            for lang in langs:
                expanded_names.append(f"{dn_lower}:{lang}")
        if not expanded_names:
            raise ValueError(
                f"No datasets left to evaluate for language {language!r}. "
                f"Requested datasets: {dataset_names}."
            )
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

        if lang:
            # Raising here happens at evaluator construction time, before any
            # (expensive) corpus encoding in __call__.
            if "language" not in queries_ds.column_names:
                raise ValueError(
                    f"A per-language evaluation ({lang!r}) was requested for "
                    f"{dataset_path}, but its queries split has no 'language' "
                    "column. Evaluating without filtering would silently score "
                    "all languages under a per-language name and corrupt the "
                    "macro averages."
                )
            queries_ds = queries_ds.filter(lambda r: r["language"] == lang)
            if len(queries_ds) == 0:
                raise ValueError(
                    f"No queries with language == {lang!r} in {dataset_path}. "
                    "Check the language spelling (full lowercase name, e.g. "
                    "'french')."
                )

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

        # The corpus prompt is the text rendered alongside each corpus image
        # (VLM processors render "<vision tokens> + text", matching
        # colpali_engine's visual_prompt_prefix). Keyed by the expanded
        # "dataset:language" names.
        document_text = None
        if self.corpus_prompts is not None:
            document_text = self.corpus_prompts.get(dataset_name)

        if document_text:
            corpus = {
                str(r[cid_col]): {"image": r["image"], "text": document_text}
                for r in corpus_ds
            }
        else:
            corpus = {str(r[cid_col]): {"image": r["image"]} for r in corpus_ds}

        # Binarize qrels (score > 0): ST's InformationRetrievalEvaluator only
        # supports binary relevance. Lossless for v1/v2 (uniform grades), but
        # v3 qrels are graded (1/2), so v3 NDCG differs from MTEB's
        # pytrec_eval graded-gain NDCG. See the class docstring.
        relevant_docs: dict[str, set[str]] = {}
        for r in qrels_ds:
            if int(r["score"]) > 0:
                qid = str(r[qrel_qid])
                if qid in queries:
                    relevant_docs.setdefault(qid, set()).add(str(r[qrel_cid]))

        human_readable_name = self._get_human_readable_name(dataset_name)

        ir_evaluator_kwargs["corpus_chunk_size"] = self._corpus_chunk_size

        # Forward the per-dataset query prompt (e.g. MTEB v3's "Find a
        # screenshot that is relevant to the user's question.") so it is
        # prepended to queries at encode time, mirroring ST's NanoBEIR.
        # The corpus prompt is deliberately NOT forwarded: it is injected as
        # the image-side "text" above, string-prefixing is meaningless for
        # image corpora.
        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(dataset_name)

        evaluator = self.information_retrieval_class(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )
        evaluator._corpus_dataset_path = dataset_path
        return evaluator

    # ── Corpus encoding ──────────────────────────────────────────────────

    @staticmethod
    def _encode_corpus(
        model: ColBERT,
        evaluator: ViDoREInformationRetrievalEvaluator,
    ) -> torch.Tensor:
        """Encode an evaluator's full corpus into a padded 3D tensor."""
        chunks: list[torch.Tensor] = []
        with (
            nullcontext()
            if evaluator.truncate_dim is None
            else model.truncate_embeddings(evaluator.truncate_dim)
        ):
            for start in trange(
                0,
                len(evaluator.corpus),
                evaluator.corpus_chunk_size,
                desc="Encoding shared corpus",
            ):
                end = min(start + evaluator.corpus_chunk_size, len(evaluator.corpus))
                corpus_chunk = evaluator._get_corpus_chunk(start, end)
                chunk_embs = torch.nn.utils.rnn.pad_sequence(
                    model.encode(
                        corpus_chunk,
                        prompt_name=evaluator.corpus_prompt_name,
                        prompt=evaluator.corpus_prompt,
                        is_query=False,
                        batch_size=evaluator.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=False,
                    ),
                    batch_first=True,
                    padding_value=0,
                )
                chunks.append(chunk_embs)

        max_tokens = max(c.shape[1] for c in chunks)
        padded = []
        for c in chunks:
            if c.shape[1] < max_tokens:
                p = torch.zeros(
                    c.shape[0],
                    max_tokens - c.shape[1],
                    c.shape[2],
                    dtype=c.dtype,
                    device=c.device,
                )
                c = torch.cat([c, p], dim=1)
            padded.append(c)

        return torch.cat(padded, dim=0)

    # ── Main entry point ───────────────────────────────────────────────

    def __call__(
        self,
        model: ColBERT,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results: dict[str, list[float]] = {}
        per_dataset_results: dict[str, float] = {}

        if epoch != -1:
            out_txt = (
                f" in epoch {epoch} after {steps} steps"
                if steps != -1
                else f" after epoch {epoch}"
            )
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(
            "ViDoRe Evaluation of the model on %s dataset%s:",
            self.dataset_names,
            out_txt,
        )

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        # Identify which corpus paths are shared across language variants
        corpus_counts: dict[str, int] = {}
        for evaluator in self.evaluators:
            path = getattr(evaluator, "_corpus_dataset_path", None)
            if path:
                corpus_counts[path] = corpus_counts.get(path, 0) + 1
        shared_paths = {p for p, c in corpus_counts.items() if c > 1}

        # Iterate evaluators, encoding each shared corpus once and passing
        # the embeddings to every language variant before freeing it.
        num_underscores_in_name = self.name.count("_")
        current_corpus_path: str | None = None
        corpus_embeddings: torch.Tensor | None = None

        for evaluator in tqdm(
            self.evaluators,
            desc="Evaluating datasets",
            disable=not self.show_progress_bar,
        ):
            logger.info("Evaluating %s", evaluator.name)
            path = getattr(evaluator, "_corpus_dataset_path", None)

            extra_kwargs: dict = {}
            if path in shared_paths:
                if path != current_corpus_path:
                    # New corpus group — free previous, encode this one once
                    del corpus_embeddings
                    logger.info(
                        "Encoding corpus for %s (%d language variants will reuse it)",
                        path,
                        corpus_counts[path],
                    )
                    corpus_embeddings = self._encode_corpus(model, evaluator)
                    current_corpus_path = path
                extra_kwargs["corpus_embeddings"] = corpus_embeddings

            evaluation = evaluator(model, output_path, epoch, steps, **extra_kwargs)

            for full_key, metric_value in evaluation.items():
                metric = full_key.split("_", maxsplit=num_underscores_in_name)[-1]
                per_metric_results.setdefault(metric, []).append(metric_value)
                per_dataset_results[full_key] = metric_value

        del corpus_embeddings

        # ── Aggregation (mirrors NanoBEIREvaluatorST) ──────────────────

        agg_results = {
            metric: self.aggregate_fn(values)
            for metric, values in per_metric_results.items()
        }

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            mode = "w" if not os.path.isfile(csv_path) else "a"
            with open(csv_path, mode=mode, encoding="utf-8") as fOut:
                if mode == "w":
                    fOut.write(",".join(self.csv_headers))
                    fOut.write("\n")

                output_data = [epoch, steps]
                for name in self.score_function_names:
                    for k in self.accuracy_at_k:
                        output_data.append(agg_results[f"{name}_accuracy@{k}"])
                    for k in self.precision_recall_at_k:
                        output_data.append(agg_results[f"{name}_precision@{k}"])
                        output_data.append(agg_results[f"{name}_recall@{k}"])
                    for k in self.mrr_at_k:
                        output_data.append(agg_results[f"{name}_mrr@{k}"])
                    for k in self.ndcg_at_k:
                        output_data.append(agg_results[f"{name}_ndcg@{k}"])
                    for k in self.map_at_k:
                        output_data.append(agg_results[f"{name}_map@{k}"])

                fOut.write(",".join(map(str, output_data)))
                fOut.write("\n")

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [
                        (
                            name,
                            agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"],
                        )
                        for name in self.score_function_names
                    ],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = (
                    f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"
                )

        avg_queries = np.mean([len(e.queries) for e in self.evaluators])
        avg_corpus = np.mean([len(e.corpus) for e in self.evaluators])
        logger.info("Average Queries: %s", avg_queries)
        logger.info("Average Corpus: %s\n", avg_corpus)

        for name in self.score_function_names:
            logger.info("Aggregated for Score Function: %s", name)
            for k in self.accuracy_at_k:
                logger.info(
                    "Accuracy@%d: %.2f%%",
                    k,
                    agg_results[f"{name}_accuracy@{k}"] * 100,
                )
            for k in self.precision_recall_at_k:
                logger.info(
                    "Precision@%d: %.2f%%",
                    k,
                    agg_results[f"{name}_precision@{k}"] * 100,
                )
                logger.info(
                    "Recall@%d: %.2f%%",
                    k,
                    agg_results[f"{name}_recall@{k}"] * 100,
                )
            for k in self.mrr_at_k:
                logger.info("MRR@%d: %.4f", k, agg_results[f"{name}_mrr@{k}"])
            for k in self.ndcg_at_k:
                logger.info("NDCG@%d: %.4f", k, agg_results[f"{name}_ndcg@{k}"])
            for k in self.map_at_k:
                logger.info("MAP@%d: %.4f", k, agg_results[f"{name}_map@{k}"])

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)

        results: dict[str, float] = per_dataset_results
        results.update(agg_results)

        # ── ViDoRe-specific aggregation ────────────────────────────────

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_underscores = self.name.count("_")

        # Per-dataset macro averages over languages (v2/v3 only)
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
                    logger.info(
                        "%s macro NDCG@%d: %.4f", base_hr, ndcg_k, results[avg_key]
                    )

        # Per-version macro averages
        version_groups: dict[str, list[str]] = {}
        for dataset_name in self.dataset_names:
            base = dataset_name.split(":")[0].lower()
            for version, datasets in VIDORE_VERSION_DATASETS.items():
                if base in datasets:
                    version_groups.setdefault(version, []).append(
                        self._get_human_readable_name(dataset_name)
                    )
                    break

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
                    logger.info(
                        "ViDoRE %s macro NDCG@%d: %.4f",
                        version,
                        ndcg_k,
                        results[ver_key],
                    )

        return results
