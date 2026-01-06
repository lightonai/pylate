from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from sentence_transformers.evaluation.InformationRetrievalEvaluator import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.util import is_datasets_available

from .pylate_information_retrieval_evaluator import PyLateInformationRetrievalEvaluator

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)

CodeSearchNetSplitType = Literal["python", "javascript", "go", "ruby", "java", "php"]


class CodeSearchNetworkEvaluator(NanoBEIREvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on the CodeSearchNet benchmark.

    The CodeSearchNet benchmark is designed for code search tasks, evaluating models on their ability
    to retrieve relevant code snippets given natural language queries. This evaluator extends the
    NanoBEIREvaluator to work with the CodeSearchNet dataset structure.

    Args:
        split_names (List[str]): The splits to evaluate on (e.g., ["python", "go"]). Defaults to ["python", "javascript", "go", "ruby", "java", "php"].
        dataset_path (str): The Hugging Face dataset path. Defaults to "lightonai/CodeSearchNet".
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to None.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all splits in split_names are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all splits in split_names are keys.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from codesearchnet_evaluator import CodeSearchNetworkEvaluator

            model = SentenceTransformer('your-code-model')

            splits = ["python", "go"]
            query_prompts = {
                "python": "Query: ",
                "go": "Query: "
            }

            evaluator = CodeSearchNetworkEvaluator(
                split_names=splits,
                query_prompts=query_prompts,
            )

            results = evaluator(model)
            print(evaluator.primary_metric)
            print(results[evaluator.primary_metric])
    """

    def __init__(
        self,
        split_names: list[CodeSearchNetSplitType] | None = None,
        dataset_path: str = "lightonai/CodeSearchNet",
        **kwargs,
    ):
        # Store dataset path before calling parent init
        self.dataset_path = dataset_path

        # Set default splits if none provided
        if split_names is None:
            split_names = ["python", "javascript", "go", "ruby", "java", "php"]

        # Override the aggregate_key default to be more descriptive
        if "aggregate_key" not in kwargs:
            kwargs["aggregate_key"] = "mean"

        # Initialize parent class with split_names as dataset_names
        # We need to bypass the parent's __init__ validation temporarily
        self.split_names = split_names

        # Call parent init with dataset_names=split_names
        super().__init__(dataset_names=split_names, **kwargs)

        # Update the name to reflect CodeSearchNet
        self.name = f"CodeSearchNet_{self.aggregate_key}"
        if self.truncate_dim:
            self.name += f"_{self.truncate_dim}"

        self.csv_file = f"CodeSearchNet_evaluation_{self.aggregate_key}_results.csv"

    def _get_human_readable_name(self, split_name: CodeSearchNetSplitType) -> str:
        """Get human-readable name for the split."""
        human_readable_name = f"CodeSearchNet{split_name.capitalize()}"
        if self.truncate_dim is not None:
            human_readable_name += f"_{self.truncate_dim}"
        return human_readable_name

    def _load_dataset(
        self, split_name: CodeSearchNetSplitType, **ir_evaluator_kwargs
    ) -> InformationRetrievalEvaluator:
        """
        Load a specific split from the CodeSearchNet dataset.

        Args:
            split_name: The name of the split to load (e.g., "python", "go")
            **ir_evaluator_kwargs: Additional arguments to pass to InformationRetrievalEvaluator

        Returns:
            InformationRetrievalEvaluator configured for the specified split
        """
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CodeSearchNetworkEvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        # Load corpus, queries, and qrels for the specific split
        corpus = load_dataset(self.dataset_path, "corpus", split=split_name)
        queries = load_dataset(self.dataset_path, "queries", split=split_name)
        qrels = load_dataset(self.dataset_path, "qrels", split=split_name)

        # Build dictionaries directly from _id fields
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        # Handle prompts
        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(
                split_name, None
            )
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(
                split_name, None
            )

        human_readable_name = self._get_human_readable_name(split_name)

        return PyLateInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            corpus_chunk_size=10,
            **ir_evaluator_kwargs,
        )

    def _validate_dataset_names(self):
        """Validate that split names are valid for CodeSearchNet."""
        valid_splits = ["python", "javascript", "go", "ruby", "java", "php"]

        if len(self.dataset_names) == 0:
            raise ValueError(
                "split_names cannot be empty. Use None to evaluate on all splits (['python', 'go'])."
            )

        if invalid_splits := [
            split_name
            for split_name in self.dataset_names
            if split_name.lower() not in valid_splits
        ]:
            raise ValueError(
                f"Split(s) {invalid_splits} not found in CodeSearchNet. "
                f"Valid split names are: {valid_splits}"
            )

        # Normalize split names to lowercase
        self.dataset_names = [split_name.lower() for split_name in self.dataset_names]

    def _validate_prompts(self):
        """Validate prompts for CodeSearchNet splits."""
        error_msg = ""

        if self.query_prompts is not None:
            if isinstance(self.query_prompts, str):
                # If string, it will be applied to all splits - no validation needed
                pass
            elif isinstance(self.query_prompts, dict):
                if missing_query_prompts := [
                    split_name
                    for split_name in self.dataset_names
                    if split_name not in self.query_prompts
                ]:
                    error_msg += f"The following splits are missing query prompts: {missing_query_prompts}\n"

        if self.corpus_prompts is not None:
            if isinstance(self.corpus_prompts, str):
                # If string, it will be applied to all splits - no validation needed
                pass
            elif isinstance(self.corpus_prompts, dict):
                if missing_corpus_prompts := [
                    split_name
                    for split_name in self.dataset_names
                    if split_name not in self.corpus_prompts
                ]:
                    error_msg += f"The following splits are missing corpus prompts: {missing_corpus_prompts}\n"

        if error_msg:
            raise ValueError(error_msg.strip())

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        """Run evaluation on CodeSearchNet splits."""
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"CodeSearchNet Evaluation of the model on {self.dataset_names} splits{out_txt}:"
        )

        # Call parent's __call__ method which handles the actual evaluation
        return super().__call__(model, output_path, epoch, steps, *args, **kwargs)
