from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from datasets import load_dataset
from pylate import evaluation, losses, models, utils

model_name = "output/answerai-colbert-small-v1"  # "distilroberta-base" # Choose the model you want
batch_size = 32  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 1

# Save path of the model
output_dir = "output/msmarco_bm25_triplet_bert-base-uncased"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

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

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    run_name="msmarco_bm25_triplet_bert-base-uncased",  # Will be used in W&B if `wandb` is installed
    learning_rate=3e-6,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)

trainer.train()
