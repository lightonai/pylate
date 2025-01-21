from __future__ import annotations

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
