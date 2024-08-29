# Available models

Here is a list of the pre-trained ColBERT models available in PyLate along with their results on BEIR:

=== "Table"

| Model                                 | BEIR AVG | NFCorpus | SciFact | SCIDOCS | FiQA2018 | TRECCOVID | HotpotQA | Touche2020 | ArguAna | ClimateFEVER | FEVER | QuoraRetrieval | NQ   | DBPedia |
|---------------------------------------|----------|----------|---------|---------|----------|-----------|----------|------------|---------|--------------|-------|----------------|------|---------|
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0)                 | 50.02    | 33.8     | 69.3    | 15.4    | 35.6     | 73.3      | 66.7     | 26.3       | 46.3    | 17.6         | 78.5  | 85.2           | 56.2 | 44.6    |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 53.79    | 37.3     | 74.77   | 18.42   | 41.15    | 84.59     | 76.11    | 25.69      | 50.09   | 33.07        | 90.96 | 87.72          | 59.1 | 45.58   |


???+ note
    `lightonai/colbertv2.0` is the original [ColBERTv2 model](https://huggingface.co/colbert-ir/colbertv2.0/tree/main) made compatible with PyLate. We thank Omar Khattab for allowing us to share the model on PyLate.

    We are planning to release various strong models in the near future, but feel free to contact us if you want to make your existing ColBERT compatible with PyLate.