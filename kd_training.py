from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from datasets import load_dataset
from giga_cherche import data_collator, losses, models, utils

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
    utils.DatasetProcessing(
        queries=queries, documents=documents
    ).add_queries_and_documents_transform,
    # remove_columns=[feature for feature in train["train"].features if "id" in feature],
)
# train = train.map(
#     utils.DatasetProcessing(
#         queries=queries, documents=documents
#     ).add_queries_and_documents,
#     # remove_columns=[feature for feature in train["train"].features if "id" in feature],
# )


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

train_loss = losses.ColBERTLossv2(model=model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    data_collator=data_collator.ColBERT(tokenize_fn=model.tokenize),
)

trainer.train()
