<div align="center">
  <h1>PyLate</h1>
  <p>Flexible Training and Retrieval for Late Interaction Models</p>
</div>

<p align="center"><img width=500 src="docs/img/logo.png"/></p>

<div align="center">
  <!-- Documentation -->
  <a href="https://lightonai.github.io/pylate/"><img src="https://img.shields.io/badge/Documentation-purple.svg?style=flat-square" alt="documentation"></a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
</div>

PyLate is a library built on top of Sentence Transformers, designed to simplify and optimize fine-tuning, inference, and retrieval with state-of-the-art ColBERT models. It enables easy fine-tuning on both single and multiple GPUs, providing flexibility for various hardware setups. PyLate also streamlines document retrieval and allows you to load a wide range of models, enabling you to construct ColBERT models from most pre-trained language models. 

## Installation

You can install PyLate using pip:

```bash
pip install pylate
```

For evaluation dependencies, use:

```bash
pip install "pylate[eval]"
```

## Documentation 

The complete documentation is available [here](https://lightonai.github.io/pylate/), which includes in-depth guides, examples, and API references.

## Datasets

PyLate supports Hugging Face [Datasets](https://huggingface.co/docs/datasets/en/index), enabling seamless triplet / knowledge distillation based training. Below is an example of creating a custom dataset for training:

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

## Training

Here’s a simple example of training a ColBERT model on the MSMARCO dataset using PyLate. This script demonstrates training with triplet loss and evaluating the model on a test set.

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

# Define the model
model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")

# Split the dataset to create a test set
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.01)

# Shuffle and select a subset of the dataset for demonstration purposes
MAX_TRAIN_SIZE, MAX_EVAL_SIZE = 100, 100
train_dataset = train_dataset.shuffle(seed=21).select(range(MAX_TRAIN_SIZE))
eval_dataset = eval_dataset.shuffle(seed=21).select(range(MAX_EVAL_SIZE))

# Define the loss function
train_loss = losses.Contrastive(model=model)

args = SentenceTransformerTrainingArguments(
    output_dir="colbert-training",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=False,  # Some GPUs support FP16 which is faster than FP32
    bf16=False,  # Some GPUs support BF16 which is a faster FP16
    # Tracking parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    learning_rate=3e-6,
)

# Evaluation procedure
dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

trainer.train()

model.save_pretrained("custom-colbert-model")
```

After training, the model can be loaded like this:

```python
from pylate import models

model = models.ColBERT(model_name_or_path="custom-colbert-model")
```

##  Retrieve

PyLate allows easy retrieval of top documents for a given query set using the trained ColBERT model and Voyager index, simply load the model and init the index:

```python
from pylate import indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
)

index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
    override=True,
)

retriever = retrieve.ColBERT(index=index)
```

Once the model and index are set up, we can add documents to the index using their embeddings and corresponding ids:

```python
documents_ids = ["1", "2", "3"]

documents = [
    "document 1 text", "document 2 text", "document 3 text"
]

# Encode the documents
documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False, # Encoding documents
    show_progress_bar=True,
)

# Add the documents ids and embeddings to the Voyager index
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

Then we can retrieve the top-k documents for a given set of queries:

```python
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True, # Encoding queries
    show_progress_bar=True,
)

scores = retriever.retrieve(
    queries_embeddings=queries_embeddings, 
    k=10,
)

print(scores)
```

Sample Output:

```python
[
    [
        {"id": "3", "score": 11.266985893249512},
        {"id": "1", "score": 10.303335189819336},
        {"id": "2", "score": 9.502392768859863},
    ],
    [
        {"id": "1", "score": 10.88800048828125},
        {"id": "3", "score": 9.950843811035156},
        {"id": "2", "score": 9.602447509765625},
    ],
]
```

## Contributing

We welcome contributions! To get started:

1. Install the development dependencies:

```bash
pip install "pylate[dev]"
```

2. Run tests:

```bash
make test
```

3. Format code with Ruff:

```bash
make ruff
```

4. Build the documentation:

```bash
make livedoc
```