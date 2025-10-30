import os

from sentence_transformers.training_args import MultiDatasetBatchSamplers

import argparse

from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from datasets import Dataset
from pylate import evaluation, losses, models, utils


def load_train_datasets():
    """Load all available splits from nomic-embed-unsupervised-data, with caching"""
    cache_dir = "./nomic_data_hf/unsupervised_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
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
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            print(f"Loaded {split}")

        except FileNotFoundError:
            print(f"{split} not found, creating dataset")
            dataset = load_dataset(
                "nomic-ai/nomic-embed-unsupervised-data", split=split
            )
            dataset.save_to_disk(f"{cache_dir}/{split}")
            
        train_dataset[split] = dataset

    # Drop last two columns for all datasets: 'dataset' and 'shard'
    for split, dataset in train_dataset.items():
        train_dataset[split] = dataset.remove_columns(["dataset", "shard"])
    return train_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train ColBERT model on unsupervised data."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default='answerdotai/ModernBERT-base',
        help="Name of the base model to use. Default is 'answerdotai/ModernBERT-base'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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
        default=0.2,
        help="Temperature of the contrastive loss.",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="If set, the model will be run in debug mode.",
    )

    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()

    # Define training parameters
    num_train_epochs = args.epochs
    lr = args.lr
    batch_size = args.bs
    temperature = args.temp

    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]
    mini_batch_size = 512 

    assert batch_size % mini_batch_size == 0, f"Batch size {batch_size} must be a multiple of mini_batch_size {mini_batch_size}."

    # Set run name and output directory
    run_name = f"unsupervised-{model_shortname}-lr{lr}-bs{batch_size}-temp{args.temp}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(model_name_or_path=model_name, document_length=180)
    # model = torch.compile(model) # It does not work on GH200 (clariden), but it does on H100 (kuma)

    # Setup evaluation and loss
    dev_evaluator = evaluation.NanoBEIREvaluator()
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=mini_batch_size,
        gather_across_devices=True,
        temperature=temperature
    )


    # Configure training arguments
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size, # ACHTUNG! When using accelerator_config.split_batches=True this is the total batch size across all GPUs, and not per GPU as the name suggests
        per_device_eval_batch_size=batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=500, # The right amount is every 500 steps for 16k batch size
        save_steps=500,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        logging_strategy="steps",
        dataloader_num_workers=4,
        seed=42,
        dataloader_drop_last=True,
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
        data_collator=utils.ColBERTCollator(model.tokenize),
    )


    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()
