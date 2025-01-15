# NanoBEIREvaluator

This class evaluates the performance of a PyLate Model on the NanoBEIR collection of datasets. This is a direct extension of the NanoBEIREvaluator from the sentence-transformers library, leveraging the PyLateInformationRetrievalEvaluator class.

The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation. The datasets are available on HuggingFace at https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6 The Evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average. Examples -------- >>> from pylate import models, evaluation >>> model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0") >>> datasets = ["SciFact"] >>> evaluator = evaluation.NanoBEIREvaluator(dataset_names=datasets) >>> results = evaluator(model) >>> results {'NanoSciFact_MaxSim_accuracy@1': 0.62, 'NanoSciFact_MaxSim_accuracy@3': 0.74, 'NanoSciFact_MaxSim_accuracy@5': 0.8, 'NanoSciFact_MaxSim_accuracy@10': 0.86, 'NanoSciFact_MaxSim_precision@1': np.float64(0.62), 'NanoSciFact_MaxSim_precision@3': np.float64(0.26666666666666666), 'NanoSciFact_MaxSim_precision@5': np.float64(0.18), 'NanoSciFact_MaxSim_precision@10': np.float64(0.096), 'NanoSciFact_MaxSim_recall@1': np.float64(0.595), 'NanoSciFact_MaxSim_recall@3': np.float64(0.715), 'NanoSciFact_MaxSim_recall@5': np.float64(0.79), 'NanoSciFact_MaxSim_recall@10': np.float64(0.85), 'NanoSciFact_MaxSim_ndcg@10': np.float64(0.7279903941189909), 'NanoSciFact_MaxSim_mrr@10': 0.6912222222222222, 'NanoSciFact_MaxSim_map@100': np.float64(0.6903374780806633), 'NanoBEIR_mean_MaxSim_accuracy@1': np.float64(0.62), 'NanoBEIR_mean_MaxSim_accuracy@3': np.float64(0.74), 'NanoBEIR_mean_MaxSim_accuracy@5': np.float64(0.8), 'NanoBEIR_mean_MaxSim_accuracy@10': np.float64(0.86), 'NanoBEIR_mean_MaxSim_precision@1': np.float64(0.62), 'NanoBEIR_mean_MaxSim_precision@3': np.float64(0.26666666666666666), 'NanoBEIR_mean_MaxSim_precision@5': np.float64(0.18), 'NanoBEIR_mean_MaxSim_precision@10': np.float64(0.096), 'NanoBEIR_mean_MaxSim_recall@1': np.float64(0.595), 'NanoBEIR_mean_MaxSim_recall@3': np.float64(0.715), 'NanoBEIR_mean_MaxSim_recall@5': np.float64(0.79), 'NanoBEIR_mean_MaxSim_recall@10': np.float64(0.85), 'NanoBEIR_mean_MaxSim_ndcg@10': np.float64(0.7279903941189909), 'NanoBEIR_mean_MaxSim_mrr@10': np.float64(0.6912222222222222), 'NanoBEIR_mean_MaxSim_map@100': np.float64(0.6903374780806633)}

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

- **score_functions** (*'dict[str, Callable[[Tensor, Tensor], Tensor]]'*) – defaults to `None`

- **main_score_function** (*'str | SimilarityFunction | None'*) – defaults to `None`

- **aggregate_fn** (*'Callable[[list[float]], float]'*) – defaults to `<function mean at 0x7fe7b9322480>`

- **aggregate_key** (*'str'*) – defaults to `mean`

- **query_prompts** (*'str | dict[str, str] | None'*) – defaults to `None`

- **corpus_prompts** (*'str | dict[str, str] | None'*) – defaults to `None`


## Attributes

- **description**

    Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification  1. Remove "Evaluator" from the class name 2. Add a space before every capital letter



## Methods

???- note "__call__"

    This is called during training to evaluate the model. It returns a score for the evaluation with a higher score indicating a better result.

    Args:     model: the model to evaluate     output_path: path where predictions and metrics are written         to     epoch: the epoch where the evaluation takes place. This is         used for the file prefixes. If this is -1, then we         assume evaluation on test data.     steps: the steps in the current epoch at time of the         evaluation. This is used for the file prefixes. If this         is -1, then we assume evaluation at the end of the         epoch.  Returns:     Either a score for the evaluation with a higher score     indicating a better result, or a dictionary with scores. If     the latter is chosen, then `evaluator.primary_metric` must     be defined

    **Parameters**

    - **model**     (*'SentenceTransformer'*)    
    - **output_path**     (*'str'*)     – defaults to `None`    
    - **epoch**     (*'int'*)     – defaults to `-1`    
    - **steps**     (*'int'*)     – defaults to `-1`    
    - **args**    
    - **kwargs**    
    
???- note "prefix_name_to_metrics"

???- note "store_metrics_in_model_card_data"

