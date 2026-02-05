import os, sys

from sentence_transformers.training_args import MultiDatasetBatchSamplers

DATA_PATH = os.environ.get("DATA_PATH")

import argparse

import torch

from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from utils import ProperFloatType
from utils import StopTrainingCallback, ContrastiveTemperatureTracker

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
    cache_dir = DATA_PATH+"nomic_data_hf/supervised_cache"
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
        type=ProperFloatType,
        default=1.0,
        help="Number of epochs to train the model. Default is 1.0. Must be <= 1.0. This does not affect the learning rate scheduler.",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="If set, the model will be run in debug mode.",
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
        "--scheduler-type",
        type=str,
        default="linear",
        help="Type of learning rate scheduler to use.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Note to include in the run name.",
    )
    parser.add_argument(
        "--prompts",
        action='store_true',
        help="If set, use prompts in the collator.",
    )
    parser.add_argument(
        "--extra-length",
        action='store_true',
        help="If set, add EXTRA_LENGTH tokens to the query and document length to compensate for prompts.",
    )
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()

    # Define training parameters
    num_train_epochs = 1 # Since we want the learning rate scheduler to always assume 1 epoch, we set this to 1. The actual number of epochs is controlled by the StopTrainingCallback.
    lr = args.lr
    batch_size = args.bs
    model_name = args.model
    if args.learnable_temperature:
        temperature = torch.nn.Parameter(torch.tensor(args.temp))
    else:
        temperature = args.temp

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
    split_batches = True
    assert batch_size % node_count == 0, "Batch size must be divisible by the number of nodes if using split_batches!"
    devices_fingerprint = f'{data_mode}-{node_count}node-{gpu_count}gpu'

    # Set run name for wandb and output directory
    model_shortname = model_name.split("/")[-1]
    run_name = f"supervised-from-{model_shortname}-lr{lr}-bs{batch_size}-temp{temperature}-" + ('-prompts' if args.prompts else '') + ('-extratokens' if (args.extra_length and not args.prompts) else '') + ('-noextratokens' if (not args.extra_length and args.prompts) else '') + (f"-{args.note}" if args.note != "" else "")
    output_dir = DATA_PATH+f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name+'/final' if os.path.exists(model_name+'/final') else model_name,
        document_length = DOCUMENT_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
        query_length = QUERY_LENGTH + (EXTRA_LENGTH if args.extra_length else 0),
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
    train_loss = losses.Contrastive(
        model=model,
        gather_across_devices=True,
        temperature=temperature
    )

    # Configure training arguments
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        # See unsupervised.py for more details on these parameters
        per_device_train_batch_size=batch_size if split_batches else batch_size // gpu_count,
        per_device_eval_batch_size=batch_size if split_batches else batch_size // gpu_count,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=1500,
        save_steps=1500,
        logging_steps=1 if args.debug else 50,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        lr_scheduler_type=args.scheduler_type,
        # warmup_ratio=0.1,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True if data_mode == "DDP" else False,
        ddp_find_unused_parameters=False,
        accelerator_config={
            "split_batches": split_batches,
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
            } if args.prompts else None)
        ),
    )

    # Stop training callback based on --epochs
    stop_at_epoch = StopTrainingCallback(args.epochs) # This does not affect the learning rate scheduler.
    trainer.add_callback(stop_at_epoch)

    # Tracking temperature
    trainer.add_callback(ContrastiveTemperatureTracker(temperature))

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
