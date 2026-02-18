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
DOCUMENT_LENGTH = 512
QUERY_PROMPT = "search_query: "
CORPUS_PROMPT = "search_document: "



def split_negatives(example, max_negatives):
    # Assuming the 'negative' column contains a list of strings
    negatives = example['negative']
    # Create new columns for each negative example
    for i, neg in enumerate(negatives):
        example[f'negative_{i}'] = neg
        if(i>=max_negatives):
            break
    return example

def get_min_negatives_count(dataset):
    return min(len(neg) for neg in dataset['negative'])

def load_train_datasets(**kwargs):
    """Load all available splits from nomic-embed-unsupervised-data, with caching"""
    cache_dir = "./data_cache/supervised"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    try:
        train_dataset = DatasetDict.load_from_disk(cache_dir)
        print("Loaded cached datasets.")
    except FileNotFoundError:
        print("No cached datasets found. Loading datasets...")
        splits = [
            'medi_sts_wiki_rephrasal',
            'fever_hn_mine',
            'msmarco_distillation_simlm_rescored_reranked_min15',
            'reddit_triples',
            'medi_sts_flickr_sampled',
            'nq_cocondensor_hn_mine_reranked_min15',
            'nli_simcse_50negs_fixed',
            'medi_supernli_sampled',
            'hotpotqa_hn_mine_shuffled',
            'medi_sts_stackexchange_dupe'
        ]

        for split in splits:
            print(f"Loading {split} dataset...")
            dataset = load_dataset(
                "nomic-ai/nomic-embed-supervised", split=split,
                **kwargs
            )

            # Map function to split negatives into separate columns
            min_negatives = get_min_negatives_count(dataset)
            print("Min negatives", min_negatives)
            dataset = dataset.map(
                lambda x: split_negatives(x, min(min_negatives, 10)),
                remove_columns=['negative']  # Remove the original 'negative' column
            )
            train_dataset[split] = dataset
            print(f"Loaded {split} dataset with {len(dataset)} examples.")
        train_dataset.save_to_disk(cache_dir)

    # Drop last two columns for all datasets: 'dataset' and 'shard'
    for split, dataset in train_dataset.items():
        train_dataset[split] = dataset.remove_columns(["dataset"])
    return train_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train ColBERT model on nomic-supervised data."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model already trained on unsupervised data."
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
        default=8e-6,
        help="Initial learning rate for training.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=64,
        help="Batch size for training. ",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.2,
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
        document_length = DOCUMENT_LENGTH + (EXTRA_LENGTH if not args.no_extra_length else 0),
        query_length = QUERY_LENGTH + (EXTRA_LENGTH if not args.no_extra_length else 0),
    )

    # Setup evaluation and loss
    evaluators_kwargs = {}
    if not args.no_prompts:
        evaluators_kwargs = {
            "query_prompts": QUERY_PROMPT,
            "corpus_prompts": CORPUS_PROMPT
        }
    dev_evaluator = evaluation.NanoBEIREvaluator(**evaluators_kwargs)
    train_loss = losses.Contrastive(
        model=model,
        gather_across_devices=True,
        temperature=temperature
    )

    # Configure training arguments
    output_dir = f"output/colbert-zero-supervised"
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        # ACHTUNG! When using accelerator_config.split_batches=True per_device_train_batch_size is the total batch size across all GPUs, and not per GPU as the name suggests
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=1500,
        save_steps=1500,
        logging_steps=50,
        fp16=False,
        bf16=True,
        learning_rate=args.lr,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
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
        data_collator=utils.ColBERTCollator(
            tokenize_fn=model.tokenize,
            prompts=({
                "query": QUERY_PROMPT,
                **{k: CORPUS_PROMPT for k in ["document", *[f"negative_{i}" for i in range(50)]]
            },
            } if not args.no_prompts else None)
        ),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
