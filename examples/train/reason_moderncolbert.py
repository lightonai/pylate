from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils


def main():
    # As ReasonIR do not re-upload the BRIGHT data, we need to load it from the original source
    def get_doc_and_ids(doc_pairs):
        doc_ids = []
        documents = []
        for dp in doc_pairs:
            doc_ids.append(str(dp["id"]))
            documents.append(dp["content"])
        return documents, doc_ids

    def process_pos_id2doc(entry, id2doc):
        pos_docs = entry["pos"]
        neg_docs = entry["neg"]
        entry["pos"] = pos_docs[0][0] + " " + id2doc[pos_docs[0][1]]
        entry["query"] = " ".join(entry["query"])
        entry["neg"] = neg_docs[0][0] + " " + neg_docs[0][1]
        return entry

    hq_dataset = load_dataset("reasonir/reasonir-data", "hq")
    bright_docs = load_dataset("xlangai/BRIGHT", "documents")
    all_docs = []
    all_ids = []
    for task in bright_docs.keys():
        docs, ids = get_doc_and_ids(bright_docs[task])
        all_docs.extend(docs)
        all_ids.extend(ids)

    id2doc = {}
    for i in range(len(all_docs)):
        id2doc[all_ids[i]] = all_docs[i]

    train_dataset = hq_dataset.map(lambda x: process_pos_id2doc(x, id2doc))

    # Define training parameters
    num_train_epochs = 3
    lr = 1e-5
    batch_size = 256
    mini_batch_size = 32
    model_name = "lightonai/GTE-ModernColBERT-v1"
    model_shortname = model_name.split("/")[-1]

    # Set run name and output directory
    run_name = f"{model_shortname}-ReasonIR"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=8192,
        query_length=128,
        skiplist_words=[],
    )

    # Setup evaluation and loss
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=mini_batch_size,
        gather_across_devices=True,
        temperature=1.0,
    )

    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        dataloader_num_workers=8,
    )

    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
