from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

train = load_dataset(
    path="./datasets/msmarco_fr_full",
    name="train",
)

queries = load_dataset(
    path="./datasets/msmarco_fr_full",
    name="queries",
)

documents = load_dataset(
    path="./datasets/msmarco_fr_full",
    name="documents",
)

train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)


model_name = "bert-base-uncased"
batch_size = 16
num_train_epochs = 1
output_dir = "output/distillation_run-bert-base"

model = models.ColBERT(model_name_or_path=model_name)

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=False,
    bf16=False,
    logging_steps=10,
    run_name="distillation_run-bert-base",
    learning_rate=1e-5,
)

train_loss = losses.Distillation(model=model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

trainer.train()
