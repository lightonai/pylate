import os, sys

import torch
from datasets import load_dataset

DATA_PATH = os.environ.get("DATA_PATH")

import argparse

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils, evaluation
from utils import ProperFloatType
from utils import StopTrainingCallback

EXTRA_LENGTH = 7  # Number of extra tokens to add to query and document length when using prompts. This is to compensate for the additional tokens added by the prompts.
QUERY_LENGTH = 32
DOCUMENT_LENGTH = 512
QUERY_PROMPT = "search_query: "
CORPUS_PROMPT = "search_document: "


def load_train_datasets(**kwargs):
    """Load the dataset for the distillation task."""
    cache_dir = DATA_PATH + "distillation/hf_cache"
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
        type=ProperFloatType,
        default=1.0,
        help="Number of epochs to train the model. Default is 1.0. Must be <= 1.0. This does not affect the learning rate scheduler.",
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
        default=16,
        help="Batch size for training. ",
    )
    parser.add_argument(
        "--ga",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Note to include in the run name.",
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
    num_train_epochs = 1  # Since we want the learning rate scheduler to always assume 1 epoch, we set this to 1. The actual number of epochs is controlled by the StopTrainingCallback.
    lr = args.lr
    batch_size = args.bs
    gradient_accumulation_steps = args.ga
    model_name = args.model

    # Configure devices and data mode
    node_count = int(os.environ.get("SLURM_NNODES", 1))
    gpu_count = torch.cuda.device_count() * node_count
    num_workers = 4
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        data_mode = "DDP"
    elif gpu_count > 1:
        data_mode = "DP"
    else:
        data_mode = "plain"
    devices_fingerprint = f'{data_mode}-{node_count}node-{gpu_count}gpu'

    # Set run name for wandb and output directory
    model_shortname = model_name.split("/")[-1]
    run_name = f"distilled-from-{model_shortname}-lr{lr}-bs{batch_size * gpu_count}-ga{gradient_accumulation_steps}" + ('-prompts' if args.prompts else '') + ('-extratokens' if (args.extra_length and not args.prompts) else '') + ('-noextratokens' if (not args.extra_length and args.prompts) else '') + (f"-{args.note}" if args.note != "" else "")
    output_dir = DATA_PATH+f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name+('/final' if model_name.startswith('/') else ''),
        document_length=DOCUMENT_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
        query_length=QUERY_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
    )
    # model = torch.compile(model)

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
        num_train_epochs=num_train_epochs,
        # Differently from unsupervised and supervised, here per_device_train_batch_size is the batch size per GPU, since we do not use accelerator_config.split_batches=True
        per_device_train_batch_size=batch_size // gpu_count,
        per_device_eval_batch_size=batch_size // gpu_count,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=int(1500 * 16 / batch_size),
        save_steps=int(1500 * 16 / batch_size),
        logging_steps=1 if args.debug else 50,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True if data_mode == "DDP" else False,
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

    # Stop training callback based on --epochs
    stop_at_epoch = StopTrainingCallback(args.epochs) # This does not affect the learning rate scheduler.
    trainer.add_callback(stop_at_epoch)

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()