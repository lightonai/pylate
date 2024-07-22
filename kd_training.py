from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from giga_cherche import data_collator, losses, models, utils

train = load_dataset(path="./msmarco_fr", name="train", cache_dir="./msmarco_fr")
queries = load_dataset(path="./msmarco_fr", name="queries", cache_dir="./msmarco_fr")
documents = load_dataset(
    path="./msmarco_fr", name="documents", cache_dir="./msmarco_fr"
)


train = train.map(
    utils.DatasetProcessing(
        queries=queries, documents=documents
    ).add_queries_and_documents,
    remove_columns=[feature for feature in train["train"].features if "id" in feature],
)


model_name = "NohTow/colbertv2_sentence_transformer"
batch_size = 2
num_train_epochs = 1
output_dir = "output/msmarco"

model = models.ColBERT(model_name_or_path=model_name)

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=False,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=10,
    run_name="colbert-st-evaluation",
    learning_rate=3e-6,
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
