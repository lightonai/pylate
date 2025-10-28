# NanoBEIREvaluator

Evaluate the performance of a PyLate Model on the NanoBEIR collection.

This is a direct extension of the NanoBEIREvaluator from the sentence-transformers library, leveraging the PyLateInformationRetrievalEvaluator class. The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before committing to a full evaluation. The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.

## Parameters

- **dataset_names** (*'list[DatasetNameType] | None'*) – defaults to `None`

- **mrr_at_k** (*'list[int]'*) – defaults to `[10]`

- **ndcg_at_k** (*'list[int]'*) – defaults to `[10]`

- **accuracy_at_k** (*'list[int]'*) – defaults to `[1, 3, 5, 10]`

- **precision_recall_at_k** (*'list[int]'*) – defaults to `[1, 3, 5, 10]`

- **map_at_k** (*'list[int]'*) – defaults to `[100]`

- **show_progress_bar** (*'bool'*) – defaults to `False`

- **batch_size** (*'int'*) – defaults to `32`

- **write_csv** (*'bool'*) – defaults to `True`

- **truncate_dim** (*'int | None'*) – defaults to `None`

- **score_functions** (*'dict[str, Callable[[Tensor, Tensor], Tensor]] | None'*) – defaults to `None`

- **main_score_function** (*'str | SimilarityFunction | None'*) – defaults to `None`

- **aggregate_fn** (*'Callable[[list[float]], float]'*) – defaults to `<function mean at 0x7ffba7dec9a0>`

- **aggregate_key** (*'str'*) – defaults to `mean`

- **query_prompts** (*'str | dict[str, str] | None'*) – defaults to `None`

- **corpus_prompts** (*'str | dict[str, str] | None'*) – defaults to `None`

- **write_predictions** (*'bool'*) – defaults to `False`


## Attributes

- **description**

    Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification  1. Replace "CE" prefix with "CrossEncoder" 2. Remove "Evaluator" from the class name 3. Add a space before every capital letter


## Examples

```python
>>> from pylate import models, evaluation

>>> model = models.ColBERT(
...     model_name_or_path="lightonai/colbertv2.0"
... )

>>> datasets = ["SciFact"]

>>> evaluator = evaluation.NanoBEIREvaluator(
...    dataset_names=datasets
... )

```

evaluator(model)
{'NanoSciFact_MaxSim_accuracy@1': 0.62, 'NanoSciFact_MaxSim_accuracy@3': 0.74, 'NanoSciFact_MaxSim_accuracy@5': 0.8, 'NanoSciFact_MaxSim_accuracy@10': 0.86, 'NanoSciFact_MaxSim_precision@1': 0.62, 'NanoSciFact_MaxSim_precision@3': 0.26666666666666666, 'NanoSciFact_MaxSim_precision@5': 0.18, 'NanoSciFact_MaxSim_precision@10': 0.096, 'NanoSciFact_MaxSim_recall@1': 0.595, 'NanoSciFact_MaxSim_recall@3': 0.715, 'NanoSciFact_MaxSim_recall@5': 0.79, 'NanoSciFact_MaxSim_recall@10': 0.85, 'NanoSciFact_MaxSim_ndcg@10': 0.7279903941189909, 'NanoSciFact_MaxSim_mrr@10': 0.6912222222222222, 'NanoSciFact_MaxSim_map@100': 0.6903374780806633, 'NanoBEIR_mean_MaxSim_accuracy@1': 0.62, 'NanoBEIR_mean_MaxSim_accuracy@3': 0.74, 'NanoBEIR_mean_MaxSim_accuracy@5': 0.8, 'NanoBEIR_mean_MaxSim_accuracy@10': 0.86, 'NanoBEIR_mean_MaxSim_precision@1': 0.62, 'NanoBEIR_mean_MaxSim_precision@3': 0.26666666666666666, 'NanoBEIR_mean_MaxSim_precision@5': 0.18, 'NanoBEIR_mean_MaxSim_precision@10': 0.096, 'NanoBEIR_mean_MaxSim_recall@1': 0.595, 'NanoBEIR_mean_MaxSim_recall@3': 0.715, 'NanoBEIR_mean_MaxSim_recall@5': 0.79, 'NanoBEIR_mean_MaxSim_recall@10': 0.85, 'NanoBEIR_mean_MaxSim_ndcg@10': 0.7279903941189909, 'NanoBEIR_mean_MaxSim_mrr@10': 0.6912222222222222, 'NanoBEIR_mean_MaxSim_map@100': 0.6903374780806633}

## Methods

???- note "__call__"

    This is called during training to evaluate the model. It returns a score for the evaluation with a higher score indicating a better result.

    Args:     model: the model to evaluate     output_path: path where predictions and metrics are written         to     epoch: the epoch where the evaluation takes place. This is         used for the file prefixes. If this is -1, then we         assume evaluation on test data.     steps: the steps in the current epoch at time of the         evaluation. This is used for the file prefixes. If this         is -1, then we assume evaluation at the end of the         epoch.  Returns:     Either a score for the evaluation with a higher score     indicating a better result, or a dictionary with scores. If     the latter is chosen, then `evaluator.primary_metric` must     be defined

    **Parameters**

    - **model**     (*'SentenceTransformer'*)
    - **output_path**     (*'str | None'*)     – defaults to `None`
    - **epoch**     (*'int'*)     – defaults to `-1`
    - **steps**     (*'int'*)     – defaults to `-1`
    - **args**
    - **kwargs**

???- note "embed_inputs"

    Call the encoder method of the model pass

    Args:     model (SentenceTransformer): Model we are evaluating     sentences (str | list[str] | np.ndarray): Text that we are embedding  Returns:     list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: The associated embedding

    **Parameters**

    - **model**     (*'SentenceTransformer'*)
    - **sentences**     (*'str | list[str] | np.ndarray'*)
    - **kwargs**

???- note "get_config_dict"

    Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.


???- note "information_retrieval_class"

    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)  Args:     queries (Dict[str, str]): A dictionary mapping query IDs to queries.     corpus (Dict[str, str]): A dictionary mapping document IDs to documents.     relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.     corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.     mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].     ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].     accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].     precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].     map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].     show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.     batch_size (int): The batch size for evaluation. Defaults to 32.     name (str): A name for the evaluation. Defaults to "".     write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.     truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.     score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function from the ``model``.     main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.     query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.     query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.     corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.     corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.     write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.         This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.  Example:     ::          import random         from sentence_transformers import SentenceTransformer         from sentence_transformers.evaluation import InformationRetrievalEvaluator         from datasets import load_dataset          # Load a model         model = SentenceTransformer('all-MiniLM-L6-v2')          # Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)         corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")         queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")         relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")          # For this dataset, we want to concatenate the title and texts for the corpus         corpus = corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])          # Shrink the corpus size heavily to only the relevant documents + 30,000 random documents         required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))         required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))         corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)          # Convert the datasets to dictionaries         corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)         queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)         relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])         for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):             qid = str(qid)             corpus_ids = str(corpus_ids)             if qid not in relevant_docs:                 relevant_docs[qid] = set()             relevant_docs[qid].add(corpus_ids)          # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.         ir_evaluator = InformationRetrievalEvaluator(             queries=queries,             corpus=corpus,             relevant_docs=relevant_docs,             name="BeIR-touche2020-subset-test",         )         results = ir_evaluator(model)         '''         Information Retrieval Evaluation of the model on the BeIR-touche2020-test dataset:         Queries: 49         Corpus: 31923          Score-Function: cosine         Accuracy@1: 77.55%         Accuracy@3: 93.88%         Accuracy@5: 97.96%         Accuracy@10: 100.00%         Precision@1: 77.55%         Precision@3: 72.11%         Precision@5: 71.43%         Precision@10: 62.65%         Recall@1: 1.72%         Recall@3: 4.78%         Recall@5: 7.90%         Recall@10: 13.86%         MRR@10: 0.8580         NDCG@10: 0.6606         MAP@100: 0.2934         '''         print(ir_evaluator.primary_metric)         # => "BeIR-touche2020-test_cosine_map@100"         print(results[ir_evaluator.primary_metric])         # => 0.29335196224364596

    **Parameters**

    - **queries**     (*'dict[str, str]'*)
    - **corpus**     (*'dict[str, str]'*)
    - **relevant_docs**     (*'dict[str, set[str]]'*)
    - **corpus_chunk_size**     (*'int'*)     – defaults to `50000`
    - **mrr_at_k**     (*'list[int]'*)     – defaults to `[10]`
    - **ndcg_at_k**     (*'list[int]'*)     – defaults to `[10]`
    - **accuracy_at_k**     (*'list[int]'*)     – defaults to `[1, 3, 5, 10]`
    - **precision_recall_at_k**     (*'list[int]'*)     – defaults to `[1, 3, 5, 10]`
    - **map_at_k**     (*'list[int]'*)     – defaults to `[100]`
    - **show_progress_bar**     (*'bool'*)     – defaults to `False`
    - **batch_size**     (*'int'*)     – defaults to `32`
    - **name**     (*'str'*)     – defaults to ``
    - **write_csv**     (*'bool'*)     – defaults to `True`
    - **truncate_dim**     (*'int | None'*)     – defaults to `None`
    - **score_functions**     (*'dict[str, Callable[[Tensor, Tensor], Tensor]] | None'*)     – defaults to `None`
    - **main_score_function**     (*'str | SimilarityFunction | None'*)     – defaults to `None`
    - **query_prompt**     (*'str | None'*)     – defaults to `None`
    - **query_prompt_name**     (*'str | None'*)     – defaults to `None`
    - **corpus_prompt**     (*'str | None'*)     – defaults to `None`
    - **corpus_prompt_name**     (*'str | None'*)     – defaults to `None`
    - **write_predictions**     (*'bool'*)     – defaults to `False`

???- note "prefix_name_to_metrics"

???- note "store_metrics_in_model_card_data"

## References

- [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)
