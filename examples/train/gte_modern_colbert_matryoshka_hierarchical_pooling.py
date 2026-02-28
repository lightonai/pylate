"""Training GTE-ModernColBERT with MatryoshkaHierarchicalPoolingLoss.

Uses learned attention-weighted pooling to iteratively halve document tokens
(N -> N/2 -> N/4 -> ...). Fully differentiable and preserves information
from all original tokens via pooling rather than selection.
"""

import os

os.environ["WANDB_PROJECT"] = "ColBERT-MTRL"
os.environ["WANDB_ENTITY"] = "lighton"

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator

from pylate import evaluation, losses, models, utils

# Load the datasets required for knowledge distillation (train, queries, documents)
train = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="train",
)

queries = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="queries",
)

documents = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="documents",
)

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Define the base model, training parameters, and output directory
model_name = "Alibaba-NLP/gte-modernbert-base"
batch_size = 16
lr = 3e-5
num_train_epochs = 1

# Set the run name for logging and output directory
run_name = (
    f"GTE-ModernColBERT-MatryoshkaHierarchicalPooling-{lr}-lr-{num_train_epochs}-epochs"
)
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ColBERT(model_name_or_path=model_name, document_length=300)

# Document token counts to train and evaluate at
n_doc_tokens = [32, 64, 128, 256]

# Create a NanoBEIR evaluator for each document token cutoff.
# The first evaluator (largest token count) determines the main score.
evaluators = []
for n_tokens in sorted(n_doc_tokens, reverse=True):
    evaluators.append(evaluation.NanoBEIREvaluator(truncate_doc_tokens=n_tokens))
dev_evaluator = SequentialEvaluator(
    evaluators, main_score_function=lambda scores: scores[0]
)

# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=5000,
    logging_steps=20,
    fp16=False,
    bf16=True,
    run_name=run_name,
    learning_rate=lr,
    warmup_ratio=0.00,
)

# Use the Distillation loss as the base loss
base_loss = losses.Distillation(model=model)

# Learned Hierarchical Pooling: iteratively halves tokens via learned
# attention-weighted pooling of consecutive pairs (N->N/2->N/4->...).
# Fully differentiable, preserves information from all original tokens.
train_loss = losses.MatryoshkaHierarchicalPoolingLoss(
    model=model,
    loss=base_loss,
    n_doc_tokens=n_doc_tokens,
)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

# Start the training process
trainer.train()
model.save_pretrained(f"{output_dir}/final")
