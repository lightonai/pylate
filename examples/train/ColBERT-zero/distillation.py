import os

from datasets import load_dataset


import argparse

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils, evaluation

def load_train_datasets():
    """Load the dataset for the distillation task."""
    cache_dir = "./distillation/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)

    train = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="train",
    )
    queries = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="queries",
    )
    documents = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="documents",
    )
   
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )
    return train

def main():
    parser = argparse.ArgumentParser(description="Run distilled training.")
    parser.add_argument(
        "--model",
        type=str,
        help="Student model to be trained",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model. Default is 1",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Initial learning rate for training.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()

    # Define training parameters
    num_train_epochs = args.epoch
    lr = args.lr
    batch_size = args.bs
    model_name = args.model
    model_shortname = model_name.split("/")[-1]

    # Set run name and output directory
    run_name = f"{model_shortname}-distilled-lr{lr}-bs{batch_size}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name+('/final' if model_name.startswith('/') else ''),
    )
    # model = torch.compile(model)

    # Setup evaluation and loss
    dev_evaluator = evaluation.NanoBEIREvaluator()
    train_loss = losses.Distillation(model=model)

    # Configure training arguments
    st_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        # See unsupervised.py for more details on these parameters
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=int(1500*16/batch_size),
        save_steps=int(1500*16/batch_size),
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        dataloader_num_workers=4,

    )

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=st_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Stop training callback based on --epochs
    stop_at_epoch = StopTrainingCallback(args.epochs) # This does not affect the learning rate scheduler.
    trainer.add_callback(stop_at_epoch)

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()