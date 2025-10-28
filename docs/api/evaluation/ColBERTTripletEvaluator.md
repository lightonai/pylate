# ColBERTTripletEvaluator

Evaluate a model based on a set of triples. The evaluation will compare the score between the anchor and the positive sample with the score between the anchor and the negative sample. The accuracy is computed as the number of times the score between the anchor and the positive sample is higher than the score between the anchor and the negative sample.



## Parameters

- **anchors** (*'list[str]'*)

    Sentences to check similarity to. (e.g. a query)

- **positives** (*'list[str]'*)

    List of positive sentences

- **negatives** (*'list[str]'*)

    List of negative sentences

- **name** (*'str'*) – defaults to ``

    Name for the output.

- **batch_size** (*'int'*) – defaults to `32`

    Batch size used to compute embeddings.

- **show_progress_bar** (*'bool'*) – defaults to `False`

    If true, prints a progress bar.

- **write_csv** (*'bool'*) – defaults to `True`

    Wether or not to write results to a CSV file.

- **truncate_dim** (*'int | None'*) – defaults to `None`

    The dimension to truncate sentence embeddings to. If None, do not truncate.


## Attributes

- **description**

    Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification  1. Replace "CE" prefix with "CrossEncoder" 2. Remove "Evaluator" from the class name 3. Add a space before every capital letter


## Examples

```python
>>> from pylate import evaluation, models

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
...     device="cpu",
... )

>>> anchors = [
...     "fruits are healthy.",
...     "fruits are healthy.",
... ]

>>> positives = [
...     "fruits are good for health.",
...     "Fruits are growing in the trees.",
... ]

>>> negatives = [
...     "Fruits are growing in the trees.",
...     "fruits are good for health.",
... ]

>>> triplet_evaluation = evaluation.ColBERTTripletEvaluator(
...     anchors=anchors,
...     positives=positives,
...     negatives=negatives,
...     write_csv=True,
... )

>>> results = triplet_evaluation(model=model, output_path=".")

>>> results
{'accuracy': 0.5}

>>> triplet_evaluation.csv_headers
['epoch', 'steps', 'accuracy']

>>> import pandas as pd
>>> df = pd.read_csv(triplet_evaluation.csv_file)
>>> assert df.columns.tolist() == triplet_evaluation.csv_headers
```

## Methods

???- note "__call__"

    Evaluate the model on the triplet dataset. Measure the scoring between the anchor and the positive with every other positive and negative samples using HITS@K.

    **Parameters**

    - **model**     (*'ColBERT'*)    
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
    
???- note "from_input_examples"

???- note "get_config_dict"

    Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.

    
???- note "prefix_name_to_metrics"

???- note "store_metrics_in_model_card_data"
