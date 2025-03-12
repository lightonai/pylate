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

<p align="justify">
PyLate is a library built on top of Sentence Transformers, designed to simplify and optimize fine-tuning, inference, and retrieval with state-of-the-art ColBERT models. It enables easy fine-tuning on both single and multiple GPUs, providing flexibility for various hardware setups. PyLate also streamlines document retrieval and allows you to load a wide range of models, enabling you to construct ColBERT models from most pre-trained language models.
</p>

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

## Training
### Contrastive training

Here’s a simple example of training a ColBERT model on the MS MARCO dataset triplet dataset using PyLate. This script demonstrates training with contrastive loss and evaluating the model on a held-out eval set:

```python
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

# Define model parameters for contrastive training
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 32  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory
run_name = "contrastive-bert-base-uncased"
output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model makes the training faster
model = torch.compile(model)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define the loss function
train_loss = losses.Contrastive(model=model)

# Initialize the evaluator
dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    learning_rate=3e-6,
)

# Initialize the trainer for the contrastive training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)
# Start the training process
trainer.train()
```


After training, the model can be loaded using the output directory path:

```python
from pylate import models

model = models.ColBERT(model_name_or_path="contrastive-bert-base-uncased")
```

### Knowledge distillation
To get the best performance when training a ColBERT model, you should use knowledge distillation to train the model using the scores of a strong teacher model.
Here's a simple example of how to train a model using knowledge distillation in PyLate on MS MARCO:
```python
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

# Load the datasets required for knowledge distillation (train, queries, documents)
train = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="train",
)

queries = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="queries",
)

documents = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="documents",
)

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Define the base model, training parameters, and output directory
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 16
num_train_epochs = 1
# Set the run name for logging and output directory
run_name = "knowledge-distillation-bert-base"
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model to make the training faster
model = torch.compile(model)

# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,
    learning_rate=1e-5,
)

# Use the Distillation loss function for training
train_loss = losses.Distillation(model=model)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

# Start the training process
trainer.train()
```



## Datasets

PyLate supports Hugging Face [Datasets](https://huggingface.co/docs/datasets/en/index), enabling seamless triplet / knowledge distillation based training. For contrastive training, you can use any of the existing sentence transformers triplet datasets. Below is an example of creating a custom triplet dataset for training:

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

To create a knowledge distillation dataset, you can use the following snippet:
```python
from datasets import Dataset

dataset = [
    {
        "query_id": 54528,
        "document_ids": [
            6862419,
            335116,
            339186,
        ],
        "scores": [
            0.4546215673141326,
            0.6575686537173476,
            0.26825184192900203,
        ],
    },
    {
        "query_id": 749480,
        "document_ids": [
            6862419,
            335116,
            339186,
        ],
        "scores": [
            0.2546215673141326,
            0.7575686537173476,
            0.96825184192900203,
        ],
    },
]


dataset = Dataset.from_list(mapping=dataset)

documents = [
    {"document_id": 6862419, "text": "example doc 1"},
    {"document_id": 335116, "text": "example doc 2"},
    {"document_id": 339186, "text": "example doc 3"},
]

queries = [
    {"query_id": 749480, "text": "example query"},
]

documents = Dataset.from_list(mapping=documents)

queries = Dataset.from_list(mapping=queries)
```

##  Retrieve

PyLate allows easy retrieval of top documents for a given query set using the trained ColBERT model and Voyager index, simply load the model and init the index:

```python
from pylate import indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="lightonai/colbertv2.0",
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

## Rerank

If you only want to use the ColBERT model to perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use rank function and pass the queries and documents to rerank:

```python
from pylate import rank

queries = [
    "query A",
    "query B",
]
documents = [
    ["document A", "document B"],
    ["document 1", "document C", "document B"],
]
documents_ids = [
    [1, 2],
    [1, 3, 2],
]

queries_embeddings = model.encode(
    queries,
    is_query=True,
)
documents_embeddings = model.encode(
    documents,
    is_query=False,
)

reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
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

## Citation

You can refer to the library with this BibTeX:

```bibtex
@misc{PyLate,
  title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
  author={Chaffin, Antoine and Sourty, Raphaël},
  url={https://github.com/lightonai/pylate},
  year={2024}
}
```
