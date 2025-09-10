# PylateModelCardData

A dataclass for storing data used in the model card.



## Parameters

- **language** (*'str | list[str] | None'*) – defaults to `<factory>`

    The model language, either a string or a list of strings, e.g., "en" or ["en", "de", "nl"].

- **license** (*'str | None'*) – defaults to `None`

    The license of the model, e.g., "apache-2.0", "mit", or "cc-by-nc-sa-4.0".

- **model_name** (*'str | None'*) – defaults to `None`

    The pretty name of the model, e.g., "SentenceTransformer based on microsoft/mpnet-base".

- **model_id** (*'str | None'*) – defaults to `None`

    The model ID for pushing the model to the Hub, e.g., "tomaarsen/sbert-mpnet-base-allnli".

- **train_datasets** (*'list[dict[str, str]]'*) – defaults to `<factory>`

    A list of dictionaries containing names and/or Hugging Face dataset IDs for training datasets, e.g., [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}].

- **eval_datasets** (*'list[dict[str, str]]'*) – defaults to `<factory>`

    A list of dictionaries containing names and/or Hugging Face dataset IDs for evaluation datasets, e.g., [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}].

- **task_name** (*'str'*) – defaults to `semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more`

    The human-readable task the model is trained on, e.g., "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".

- **tags** (*'list[str] | None'*) – defaults to `<factory>`

    A list of tags for the model, e.g., ["sentence-transformers", "sentence-similarity", "feature-extraction"].

- **generate_widget_examples** (*"Literal['deprecated']"*) – defaults to `deprecated`


## Attributes

- **base_model**

- **base_model_revision**

- **best_model_step**

- **code_carbon_callback**

- **license**

- **model**

- **model_id**

- **model_name**

- **predict_example**

- **trainer**



## Methods

???- note "add_tags"

???- note "compute_dataset_metrics"

    Given a dataset, compute the following: * Dataset Size * Dataset Columns * Dataset Stats     - Strings: min, mean, max word count/token length     - Integers: Counter() instance     - Floats: min, mean, max range     - List: number of elements or min, mean, max number of elements * 3 Example samples * Loss function name     - Loss function config

    **Parameters**

    - **dataset**     (*'Dataset | IterableDataset | None'*)
    - **dataset_info**     (*'dict[str, Any]'*)
    - **loss**     (*'dict[str, nn.Module] | nn.Module | None'*)

???- note "extract_dataset_metadata"

???- note "format_eval_metrics"

    Format the evaluation metrics for the model card.

    The following keys will be returned: - eval_metrics: A list of dictionaries containing the class name, description, dataset name, and a markdown table   This is used to display the evaluation metrics in the model card. - metrics: A list of all metric keys. This is used in the model card metadata. - model-index: A list of dictionaries containing the task name, task type, dataset type, dataset name, metric name,   metric type, and metric value. This is used to display the evaluation metrics in the model card metadata.


???- note "format_training_logs"

???- note "get"

    Get value for a given metadata key.

    **Parameters**

    - **key**     (*str*)
    - **default**     (*Any*)     – defaults to `None`

???- note "get_codecarbon_data"

???- note "get_model_specific_metadata"

???- note "infer_datasets"

???- note "pop"

    Pop value for a given metadata key.

    **Parameters**

    - **key**     (*str*)
    - **default**     (*Any*)     – defaults to `None`

???- note "register_model"

???- note "set_base_model"

???- note "set_best_model_step"

???- note "set_evaluation_metrics"

???- note "set_label_examples"

???- note "set_language"

???- note "set_license"

???- note "set_losses"

???- note "set_model_id"

???- note "set_widget_examples"

    A function to create widget examples from a dataset. For now, set_widget_examples is not compatible with our transform/map operations, so we make it a no-op until it is fixed

    **Parameters**

    - **dataset**     (*'Dataset | DatasetDict'*)

???- note "to_dict"

    Converts CardData to a dict.

    Returns:     `dict`: CardData represented as a dictionary ready to be dumped to a YAML     block for inclusion in a README.md file.


???- note "to_yaml"

    Dumps CardData to a YAML block for inclusion in a README.md file.

    Args:     line_break (str, *optional*):         The line break to use when dumping to yaml.  Returns:     `str`: CardData represented as a YAML block.

    **Parameters**

    - **line_break**     – defaults to `None`

???- note "tokenize"

???- note "try_to_set_base_model"

???- note "validate_datasets"
