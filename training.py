from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from giga_cherche import data_collator, evaluation, losses, models

model_name = "NohTow/colbertv2_sentence_transformer"  # "distilroberta-base" # Choose the model you want
batch_size = 32  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 1

# Save path of the model
output_dir = f"output/msmarco_{model_name.replace('/', '-')}_{batch_size}_bs_{num_train_epochs}_epoch"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]


# Subsample the training dataset
MAX_EXAMPLES = 100000
train_dataset = train_dataset.shuffle(seed=21).select(range(MAX_EXAMPLES))

train_loss = losses.ColBERTLossv1(model=model)

# Subsample the evaluation dataset
# max_samples = 1000
# eval_dataset = eval_dataset.select(range(max_samples))

# Initialize the evaluator
dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    # name=f"msmarco-bm25",
)

# Eval base model
# dev_evaluator(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=10,
    run_name="colbert-st-evaluation",  # Will be used in W&B if `wandb` is installed
    # report_to="wandb",
    learning_rate=3e-6,
    # gradient_accumulation_steps=8,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=data_collator.ColBERT(model.tokenize),
)

trainer.train()
