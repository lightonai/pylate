# ColBERTDistillationEvaluator

ColBERT Distillation Evaluator. This class is used to monitor the distillation process of a ColBERT model.



## Parameters

- **queries** (*'list[str]'*)

    Set of queries.

- **documents** (*'list[list[str]]'*)

    Set of documents. Each query has a list of documents. Each document is a list of strings. Number of documents should be the same for each query.

- **scores** (*'list[list[float]]'*)

    The scores associated with the documents. Each query / documents pairs has a list of scores.

- **name** (*'str'*) – defaults to ``

    The name of the evaluator.

- **batch_size** (*'int'*) – defaults to `16`

    The batch size.

- **show_progress_bar** (*'bool'*) – defaults to `False`

    Whether to show the progress bar.

- **write_csv** (*'bool'*) – defaults to `True`

    Whether to write the results to a CSV file.

- **truncate_dim** (*'int | None'*) – defaults to `None`

    The dimension to truncate the embeddings.

- **normalize_scores** (*'bool'*) – defaults to `True`


## Attributes

- **description**

    Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification  1. Replace "CE" prefix with "CrossEncoder" 2. Remove "Evaluator" from the class name 3. Add a space before every capital letter


## Examples

```python
>>> from pylate import models, evaluation

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
... )

>>> queries = [
...     "query A",
...     "query B",
... ]

>>> documents = [
...     ["document A", "document B", "document C"],
...     ["document C C", "document B B", "document A A"],
... ]

>>> scores = [
...     [0.9, 0.1, 0.05],
...     [0.05, 0.9, 0.1],
... ]

>>> distillation_evaluator = evaluation.ColBERTDistillationEvaluator(
...     queries=queries,
...     documents=documents,
...     scores=scores,
...     write_csv=True,
... )

>>> results = distillation_evaluator(model=model, output_path=".")

>>> assert "kl_divergence" in results
>>> assert isinstance(results["kl_divergence"], float)

>>> import pandas as pd
>>> df = pd.read_csv(distillation_evaluator.csv_file)
>>> assert df.columns.tolist() == distillation_evaluator.csv_headers
```

## Methods

???- note "__call__"

    This is called during training to evaluate the model. It returns a score for the evaluation with a higher score indicating a better result.

    Args:     model: the model to evaluate     output_path: path where predictions and metrics are written         to     epoch: the epoch where the evaluation takes place. This is         used for the file prefixes. If this is -1, then we         assume evaluation on test data.     steps: the steps in the current epoch at time of the         evaluation. This is used for the file prefixes. If this         is -1, then we assume evaluation at the end of the         epoch.  Returns:     Either a score for the evaluation with a higher score     indicating a better result, or a dictionary with scores. If     the latter is chosen, then `evaluator.primary_metric` must     be defined

    **Parameters**

    - **model**     (*"'SentenceTransformer'"*)    
    - **output_path**     (*'str'*)     – defaults to `None`    
    - **epoch**     (*'int'*)     – defaults to `-1`    
    - **steps**     (*'int'*)     – defaults to `-1`    
    
???- note "embed_inputs"

    Call the encoder method of the model pass

    Args:     model (SentenceTransformer): Model we are evaluating     sentences (str | list[str] | np.ndarray): Text that we are embedding  Returns:     list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]: The associated embedding

    **Parameters**

    - **model**     (*'SentenceTransformer'*)    
    - **sentences**     (*'str | list[str] | np.ndarray'*)    
    - **kwargs**    
    
???- note "get_config_dict"

    Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.

    
???- note "prefix_name_to_metrics"

???- note "store_metrics_in_model_card_data"
