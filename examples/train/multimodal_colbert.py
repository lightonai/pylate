"""Example script for training a multimodal ColBERT model (ColPali/ColQwen style).

This demonstrates visual document retrieval where queries are text and documents are images,
using a Vision-Language Model (VLM) as the backbone.
"""

from __future__ import annotations

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

# Define the base VLM model
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Replace with your VLM of choice
batch_size = 4  # VLMs require smaller batch sizes due to memory
num_train_epochs = 1
run_name = "multimodal-colbert-qwen2vl"
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from a VLM
# processor_kwargs controls image processing parameters
model = models.ColBERT(
    model_name_or_path=model_name,
    processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
    document_length=1024,  # VLMs produce more tokens per image
    trust_remote_code=True,
)

# Load a multimodal dataset with text queries and image documents
# The dataset should have columns like: "query" (str), "positive" (PIL Image), "negative" (PIL Image)
# Example: dataset = load_dataset("your-org/visual-retrieval-dataset", split="train")

# For demonstration, we show the expected format:
# Each row should contain:
#   - "query": a text query string
#   - "positive": a PIL Image (the relevant document image)
#   - "negative": a PIL Image (an irrelevant document image)

# Define the loss function
train_loss = losses.Contrastive(model=model)

# Configure training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=False,
    bf16=True,
    run_name=run_name,
    learning_rate=2e-5,
    logging_steps=10,
)

# The collator handles mixed-modality batches:
# - Text query columns are tokenized with prefix insertion and query expansion
# - Image document columns are processed through the VLM processor
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     loss=train_loss,
#     data_collator=utils.ColBERTCollator(preprocess_fn=model.preprocess),
# )

# trainer.train()
# model.save_pretrained(f"{output_dir}/final")

print("Multimodal ColBERT example setup complete.")
print(f"Model supports modalities: {model.modalities}")
print("To train, uncomment the trainer section and provide a multimodal dataset.")
