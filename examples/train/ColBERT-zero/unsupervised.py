import os
from sentence_transformers.training_args import MultiDatasetBatchSamplers
import argparse
import torch
from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import evaluation, losses, models, utils

EXTRA_LENGTH = 7  # Number of extra tokens to add to query and document length when using prompts. This is to compensate for the additional tokens added by the prompts.
QUERY_LENGTH = 32
DOCUMENT_LENGTH = 180
QUERY_PROMPT = "search_query: "
CORPUS_PROMPT = "search_document: "


def load_train_datasets(**kwargs):
    """Load all available splits from nomic-embed-unsupervised-data, with caching."""
    cache_dir = "./data_cache/unsupervised"
    os.makedirs(cache_dir, exist_ok=True)
    print("Cache directory:", cache_dir)
    train_dataset = DatasetDict()
    try:
        train_dataset = DatasetDict.load_from_disk(cache_dir)
        print("Loaded cached datasets.")
    except FileNotFoundError:
        print("No cached datasets found. Loading datasets...")
        splits = [
            "reddit_title_body",
            "amazon_reviews",
            "paq",
            "s2orc_citation_titles",
            "s2orc_title_abstract",
            "s2orc_abstract_citation",
            "s2orc_abstract_body",
            "wikianswers",
            "wikipedia",
            "gooaq",
            "codesearch",
            "yahoo_title_answer",
            "agnews",
            "amazonqa",
            "yahoo_qa",
            "yahoo_title_question",
            "ccnews",
            "npr",
            "eli5",
            "cnn",
            "stackexchange_duplicate_questions",
            "stackexchange_title_body",
            "stackexchange_body_body",
            "sentence_compression",
            "wikihow",
            "altlex",
            "quora",
            "simplewiki",
            "squad",
        ]

        for split in splits:
            print(f"Loading {split} dataset...")
            # data_files = {split: f"data/{split}-*"}
            dataset = load_dataset(
                "nomic-ai/nomic-embed-unsupervised-data", split=split,
                **kwargs,
            )
            train_dataset[split] = dataset
            print(f"Loaded {split} dataset with {len(dataset)} examples.")
        train_dataset.save_to_disk(cache_dir)

    # Drop last two columns for all datasets: 'dataset' and 'shard'
    for split, dataset in train_dataset.items():
        train_dataset[split] = dataset.remove_columns(["dataset", "shard"])
    return train_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train ColBERT model on nomic-unsupervised data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default='answerdotai/ModernBERT-base',
        help="Base model to be trained.",
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
        default=3e-4,
        help="Initial learning rate for training.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=16384,
        help="Batch size for training. It must be a multiple of 512 and the number of GPUs!",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.02,
        help="Temperature of the contrastive loss.",
    )
    parser.add_argument(
        "--learnable-temperature",
        action='store_true',
        help="If set, the temperature of the contrastive loss will be learned.",
    )
    parser.add_argument(
        "--no-prompts",
        action='store_true',
        help="If set, do not use prompts in the collator.",
    )
    parser.add_argument(
        "--no-extra-length",
        action='store_true',
        help="If set, do not add extra length to the query and document length for the prompts.",
    )
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()

    # Define training parameters
    if args.learnable_temperature:
        temperature = torch.nn.Parameter(torch.tensor(args.temp))
    else:
        temperature = args.temp

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=DOCUMENT_LENGTH + (EXTRA_LENGTH if not args.no_extra_length else 0),
        query_length=QUERY_LENGTH + (EXTRA_LENGTH if not args.no_extra_length else 0),
    )

    # Setup evaluation and loss
    evaluators_kwargs = {}
    if not args.no_prompts:
        evaluators_kwargs = {
            "query_prompts": QUERY_PROMPT,
            "corpus_prompts": CORPUS_PROMPT
        }
    dev_evaluator = evaluation.NanoBEIREvaluator(**evaluators_kwargs)
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=512, # This is for H100/ GH200. Change this to fit your GPU if you are using a different one. 
        gather_across_devices=True,
        temperature=temperature
    )

    # Configure training arguments
    output_dir = f"output/colbert-zero-unsupervised"
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        # ACHTUNG! When using accelerator_config.split_batches=True per_device_train_batch_size is the total batch size across all GPUs, and not per GPU as the name suggests
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        fp16=False,
        bf16=True,
        learning_rate=args.lr,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        # split_batches=True ensures all GPUs sample from the same sub-dataset chosen by MultiDatasetBatchSamplers.PROPORTIONAL. This is required for contrastive loss to effectively find hard negatives within the same dataset across all ranks.
        accelerator_config={
            "split_batches": True,
        },
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
                "document": CORPUS_PROMPT
            } if not args.no_prompts else None)
        ),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()
