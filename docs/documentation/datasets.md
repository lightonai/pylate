PyLate is designed to be compatible with Hugging Face datasets, facilitating seamless integration for tasks such as knowledge distillation and contrastive model training. Below are examples illustrating how to load and prepare datasets for these specific training objectives.


## Constrastive dataset

Contrastive training involves datasets that contain a query, a positive document (relevant to the query), and a negative document (irrelevant to the query). This the standard triplet format used by Sentence Transformers and so PyLate contrastive training is **compatible with every existing triplet datasets.**

### Loading a pre-built contrastive dataset

You can directly download an existing contrastive dataset from Hugging Face's hub, such as the [msmarco-bm25 triplet dataset](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25).

```python
from datasets import load_dataset

dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")

train_dataset, test_dataset = dataset.train_test_split(test_size=0.001)
```

Then we can shuffle the dataset:

```python
train_dataset = train_dataset.shuffle(seed=42)
```

And select a subset of the dataset if needed:

```python
train_dataset = train_dataset.select(range(10_000))
```

### Creating a contrastive dataset from list

If you want to create a custom contrastive dataset, you can do so by manually specifying the query, positive, and negative samples.

```python
from datasets import Dataset

dataset = [
    {
        "query": "example query 1",
        "positive": "example positive document 1",
        "negative": "example negative document 1",
    },
    {
        "query": "example query 2",
        "positive": "example positive document 2",
        "negative": "example negative document 2",
    },
    {
        "query": "example query 3",
        "positive": "example positive document 3",
        "negative": "example negative document 3",
    },
]

dataset = Dataset.from_list(mapping=dataset)

train_dataset, test_dataset = dataset.train_test_split(test_size=0.3)
```

### Loading a contrastive dataset from a local parquet file

To load a local dataset stored in a Parquet file:

```python
from datasets import load_dataset

dataset = load_dataset(
    path="parquet", 
    data_files="dataset.parquet", 
    split="train"
)

train_dataset, test_dataset = dataset.train_test_split(test_size=0.001)
```



## Knowledge distillation dataset

For fine-tuning a model using knowledge distillation loss, three distinct dataset files are required: train, queries, and documents. Each file contains unique and complementary information necessary for the distillation process:


- `train`: Contains three columns: `['query_id', 'document_ids', 'scores']`
    - `query_id` refers to the query identifier.
    - `document_ids` is a list of document IDs relevant to the query.
    - `scores` corresponds to the relevance scores between the query and each document.

Example entry:

```python
{
    "query_id": 54528,
    "document_ids": [
        6862419,
        335116,
        339186,
        7509316,
        7361291,
        7416534,
        5789936,
        5645247,
    ],
    "scores": [
        0.4546215673141326,
        0.6575686537173476,
        0.26825184192900203,
        0.5256195579370395,
        0.879939718687207,
        0.7894968184862693,
        0.6450100468854655,
        0.5823844608171467,
    ],
}
```

Note: Ensure that the length of `document_ids` matches the length of `scores`.

- `queries`: Contains two columns: `['query_id', 'text']`

Example entry:

```python
{"query_id": 749480, "text": "what is function of magnesium in human body"}
```

- `documents`: contains two columns: `['document_ids', 'text']`

Example entry:

```python
{
    "document_id": 136062,
    "text": "Document text",
}
```
### Loading a pre-built knowledge distillation dataset
You can directly download an existing knowledge distillation dataset from Hugging Face's hub, such as the [MS MARCO dataset with BGE M3 scores](https://huggingface.co/datasets/lightonai/ms-marco-en-bge).
Simply load the different files by giving the respective names to the ```load_dataset``` function:

```python
from datasets import load_dataset

train = load_dataset(
    "lightonai/ms-marco-en-bge",
    "train",
    split="train",
)

queries = load_dataset(
    "lightonai/ms-marco-en-bge",
    "queries",
    split="train",
)

documents = load_dataset(
    "lightonai/ms-marco-en-bge",
    "documents",
    split="train",
)
```
### Knowledge distillation dataset from list

You can also create custom datasets from list in Python. This example demonstrates how to build the `train`, `queries`, and `documents` datasets

```python
from datasets import Dataset

dataset = [
    {
        "query_id": 54528,
        "document_ids": [
            6862419,
            335116,
            339186,
            7509316,
            7361291,
            7416534,
            5789936,
            5645247,
        ],
        "scores": [
            0.4546215673141326,
            0.6575686537173476,
            0.26825184192900203,
            0.5256195579370395,
            0.879939718687207,
            0.7894968184862693,
            0.6450100468854655,
            0.5823844608171467,
        ],
    },
    {
        "query_id": 749480,
        "document_ids": [
            6862419,
            335116,
            339186,
            7509316,
            7361291,
            7416534,
            5789936,
            5645247,
        ],
        "scores": [
            0.2546215673141326,
            0.7575686537173476,
            0.96825184192900203,
            0.0256195579370395,
            0.779939718687207,
            0.2894968184862693,
            0.1450100468854655,
            0.7823844608171467,
        ],
    },
]


dataset = Dataset.from_list(mapping=dataset)

documents = [
    {"document_id": 6862419, "text": "Document text"},
    {"document_id": 335116, "text": "Document text"},
    {"document_id": 339186, "text": "Document text"},
    {"document_id": 7509316, "text": "Document text"},
    {"document_id": 7361291, "text": "Document text"},
    {"document_id": 7416534, "text": "Document text"},
    {"document_id": 5789936, "text": "Document text"},
    {"document_id": 5645247, "text": "Document text"},
]

queries = [
    {"query_id": 749480, "text": "what is function of magnesium in human body"},
    {"query_id": 54528, "text": "what is the capital of France"},
]

documents = Dataset.from_list(mapping=documents)

queries = Dataset.from_list(mapping=queries)
```

