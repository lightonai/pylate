# Available models

!!! tip
    Following an update, all the models trained using the stanford-nlp ColBERT library or RAGatouille should be compatible with PyLate natively (including their configurations).
    You can simply load the model in PyLate:

    ```python
    from pylate import models

    model = models.ColBERT(
        model_name_or_path="colbert-ir/colbertv2.0",
    )
    ```
    or
    ```python
    model = models.ColBERT(
        model_name_or_path="jinaai/jina-colbert-v2",
        trust_remote_code=True,
    )
    ```


Here is a list of some of the pre-trained ColBERT models available in PyLate along with their results on BEIR:

=== "Table"

| Model                                 | BEIR AVG | NFCorpus | SciFact | SCIDOCS | FiQA2018 | TRECCOVID | HotpotQA | Touche2020 | ArguAna | ClimateFEVER | FEVER | QuoraRetrieval | NQ   | DBPedia |
|---------------------------------------|----------|----------|---------|---------|----------|-----------|----------|------------|---------|--------------|-------|----------------|------|---------|
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0)                 | 50.02    | 33.8     | 69.3    | 15.4    | 35.6     | 73.3      | 66.7     | 26.3       | 46.3    | 17.6         | 78.5  | 85.2           | 56.2 | 44.6    |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 53.79    | 37.3     | 74.77   | 18.42   | 41.15    | 84.59     | 76.11    | 25.69      | 50.09   | 33.07        | 90.96 | 87.72          | 59.1 | 45.58   |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 53.1    | 34.6     | 67.8   | 18.6   | 40.8    | 83.4     | 76.6    | 27.4      | 36.6   | 23.9        | 80.05 | 88.7          | 64.0 | 47.1   |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 54.89    | 37.93     | 76.34   | 19.06   | 48.51    | 83.59     | 77.32    | 31.23      | 48.51   | 30.62       | 87.44 | 86.61          | 61.8 | 48.3   |



???+ note
    `lightonai/colbertv2.0` is the original [ColBERTv2 model](https://huggingface.co/colbert-ir/colbertv2.0/tree/main) made compatible with PyLate before we supported loading directly model from Stanford-NLP. We thank Omar Khattab for allowing us to share the model on PyLate.
