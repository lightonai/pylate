# PyLateInformationRetrievalEvaluator

This class evaluates an Information Retrieval (IR) setting. This is a direct extension of the InformationRetrievalEvaluator from the sentence-transformers library, only override the compute_metrices method to be compilatible with PyLate models (define assymetric encoding using is_query params and add padding).



## Parameters

- **queries** (*'dict[str, str]'*)

- **corpus** (*'dict[str, str]'*)

- **relevant_docs** (*'dict[str, set[str]]'*)

- **corpus_chunk_size** (*'int'*) – defaults to `50000`

- **mrr_at_k** (*'list[int]'*) – defaults to `[10]`

- **ndcg_at_k** (*'list[int]'*) – defaults to `[10]`

- **accuracy_at_k** (*'list[int]'*) – defaults to `[1, 3, 5, 10]`

- **precision_recall_at_k** (*'list[int]'*) – defaults to `[1, 3, 5, 10]`

- **map_at_k** (*'list[int]'*) – defaults to `[100]`

- **show_progress_bar** (*'bool'*) – defaults to `False`

- **batch_size** (*'int'*) – defaults to `32`

- **name** (*'str'*) – defaults to ``

- **write_csv** (*'bool'*) – defaults to `True`

- **truncate_dim** (*'int | None'*) – defaults to `None`

- **score_functions** (*'dict[str, Callable[[Tensor, Tensor], Tensor]] | None'*) – defaults to `None`

- **main_score_function** (*'str | SimilarityFunction | None'*) – defaults to `None`

- **query_prompt** (*'str | None'*) – defaults to `None`

- **query_prompt_name** (*'str | None'*) – defaults to `None`

- **corpus_prompt** (*'str | None'*) – defaults to `None`

- **corpus_prompt_name** (*'str | None'*) – defaults to `None`

- **write_predictions** (*'bool'*) – defaults to `False`


## Attributes

- **description**

    Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification  1. Replace "CE" prefix with "CrossEncoder" 2. Remove "Evaluator" from the class name 3. Add a space before every capital letter



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
    
???- note "compute_dcg_at_k"

???- note "compute_metrices"

???- note "compute_metrics"

???- note "embed_inputs"

    Call the encoder method of the model pass

    Args:     model (SentenceTransformer): Model we are evaluating     sentences (str | list[str] | np.ndarray): Text that we are embedding  Returns:     list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: The associated embedding

    **Parameters**

    - **model**     (*'SentenceTransformer'*)    
    - **sentences**     (*'str | list[str] | np.ndarray'*)    
    - **encode_fn_name**     (*'str | None'*)     – defaults to `None`    
    - **prompt_name**     (*'str | None'*)     – defaults to `None`    
    - **prompt**     (*'str | None'*)     – defaults to `None`    
    - **kwargs**    
    
???- note "get_config_dict"

    Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.

    
???- note "output_scores"

???- note "prefix_name_to_metrics"

???- note "store_metrics_in_model_card_data"

