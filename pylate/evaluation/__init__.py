from __future__ import annotations

from .beir import evaluate, get_beir_triples, load_beir
from .colbert_distillation import ColBERTDistillationEvaluator
from .colbert_triplet import ColBERTTripletEvaluator
from .custom_dataset import load_custom_dataset
from .nano_beir_evaluator import NanoBEIREvaluator
from .pylate_information_retrieval_evaluator import PyLateInformationRetrievalEvaluator

try:
    from .vidore_evaluator import ViDoREvaluator
except ImportError:
    pass

__all__ = [
    "ColBERTTripletEvaluator",
    "ColBERTDistillationEvaluator",
    "NanoBEIREvaluator",
    "ViDoREvaluator",
    "get_beir_triples",
    "load_beir",
    "load_custom_dataset",
    "evaluate",
    "PyLateInformationRetrievalEvaluator",
]
