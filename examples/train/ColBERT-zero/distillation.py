import os, sys
import torch
from datasets import load_dataset
import argparse
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import losses, models, utils, evaluation

EXTRA_LENGTH = 7  # Number of extra tokens to add to query and document length when using prompts. This is to compensate for the additional tokens added by the prompts.
QUERY_LENGTH = 32
DOCUMENT_LENGTH = 512
QUERY_PROMPT = "search_query: "
CORPUS_PROMPT = "search_document: "

GRADIENT_ACCUMULATION_STEPS = 2


def load_train_datasets(**kwargs):
    """Load the dataset for the distillation task."""
    cache_dir = "./data_cache/distillation"
    os.makedirs(cache_dir, exist_ok=True)
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
    return train

def main():
    parser = argparse.ArgumentParser(
        description="Train ColBERT model on LightOn MS MARCO distilled from Gemma model."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Student model to be trained",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of epochs to train the model. Default is 1.0.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-6,
        help="Initial learning rate for training.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size for training. Note that this INCLUDES the gradient accumulation steps, so the effective batch size for a single step is bs / gradient_accumulation_steps, accumulated for gradient_accumulation_steps steps. ",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="If set, the model will be run in debug mode.",
    )
    parser.add_argument(
        "--prompts",
        action='store_true',
        help="If set, use prompts in the collator.",
    )
    parser.add_argument(
        "--extra-length",
        action='store_true',
        help="If set, use extra length for query and document.",
    )
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()

    # Define training parameters
    lr = args.lr
    batch_size = args.bs
    assert batch_size % torch.cuda.device_count() == 0, "Batch size must be a multiple of the number of GPUs!"
    assert batch_size % GRADIENT_ACCUMULATION_STEPS == 0, "Batch size must be a multiple of the gradient accumulation steps!"
    per_device_batch_size = batch_size // torch.cuda.device_count() // GRADIENT_ACCUMULATION_STEPS
    model_name = args.model
    output_dir = f"output/colbert-zero-distillation"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name+('/final' if model_name.startswith('/') else ''),
        document_length=DOCUMENT_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
        query_length=QUERY_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
    )

    # Setup evaluation and loss
    evaluators_kwargs = {}
    if args.prompts:
        evaluators_kwargs = {
            "query_prompts": QUERY_PROMPT,
            "corpus_prompts": CORPUS_PROMPT
        }
    dev_evaluator = evaluation.NanoBEIREvaluator(**evaluators_kwargs)
    train_loss = losses.Distillation(model=model)

    # Configure training arguments
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        # Differently from unsupervised and supervised, here per_device_train_batch_size is the batch size per GPU, since we do not use accelerator_config.split_batches=True
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="steps",
        eval_steps=int(1500 * 16 / batch_size),
        save_steps=int(1500 * 16 / batch_size),
        logging_steps=1 if args.debug else 50,
        fp16=False,
        bf16=True,
        learning_rate=lr,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_drop_last=True, # Needed for DDP
        ddp_find_unused_parameters=False,
    )

    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=st_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(
            tokenize_fn=model.tokenize,
            prompts=({
                "query": QUERY_PROMPT,
                "documents": CORPUS_PROMPT
            } if args.prompts else None)
        ),
    )
    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()