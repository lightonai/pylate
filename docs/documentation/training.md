# ColBERT Training
PyLate training is based on Sentence Transformer (and thus transformers) trainer, enabling a lot of functionnality such multi-GPU and FP16/BF16 training as well as logging to Weights & Biases out-of-the-box. This allows efficient, scalable and monitorable training. There are two primary ways to train ColBERT models using PyLate:

1. **Contrastive Loss (Simplest Method)**: The easiest way to train your model is by using contrastive loss, which only requires a dataset containing triplets—each consisting of a query, a positive document (relevant to the query), and a negative document (irrelevant to the query). This method trains the model to maximize the similarity between the query and the positive document, while minimizing it with the negative document.

2. **Knowledge Distillation**: To train a ColBERT model using knowledge distillation, you need to provide a dataset with three components: queries, documents, and the relevance scores between them. This method compresses the knowledge of a larger model / more accurate model (cross-encoder) into a smaller one, using the relevance scores to guide the training process.

## Contrastive Training
The original training of ColBERT was done using contrastive learning, that is, train the model to differentiate between relevant (positive) and irrelevant (negative) documents for a given query by maximizing the similarity between a query and a positive document while minimizing the similarity with irrelevant documents.

The contrastive learning in PyLate is done using triplet dataset, that is, a query is associated to one positive and one negative. It is thus **compatible with any triplet datasets from the sentence-transformers library**.

During training, the model is tasked to maximize the similarity of the query with its positive while minimizing the similarity with all the negatives as well as the positives of the other queries in the batch (thus also leveraging in-batch negatives).

Here is a example of code to run contrastive training using PyLate:

```python
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from datasets import load_dataset
from pylate import evaluation, losses, models, utils

# Define model parameters for contrastive training
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use
batch_size = 32  # A larger batch size often improves results, but requires more GPU memory
num_train_epochs = 1  # Adjust based on your requirements

# Set the output directory for saving the trained model
output_dir = "output/msmarco_bm25_contrastive_bert-base-uncased"

# Initialize the ColBERT model, adding a linear layer if it's not already a ColBERT model
model = models.ColBERT(model_name_or_path=model_name)

# Load the contrastive dataset (query, positive, and negative pairs)
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")

# Split the dataset into training and evaluation subsets
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define the contrastive loss function for training
train_loss = losses.Contrastive(model=model)

# Set up an evaluator for validation using the contrastive approach (query, positive, negative)
dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=False,  # Disable FP16 if the GPU does not support it
    bf16=True,   # Enable BF16 if supported by the GPU
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=10,
    report_to="none",  # Set to 'none' to avoid sending data to monitoring services like W&B
    run_name="msmarco_bm25_contrastive_bert-base-uncased",
    learning_rate=3e-6,  # Adjust learning rate based on the task
)

# Initialize the trainer for the contrastive training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)

# Start the training process
trainer.train()

```

## Knowledge Distillation Training

The training of late-interaction models have shown to benefit from knowledge distillation compared to a more simple contrastive learning.
Knowledge distillation training aim at making ColBERT models learn to reproduce the outputs of a more capable (e.g, a cross-encoder) teacher model. This is done by using a dataset containing queries, documents and the scores attributed by the teacher to the different query/document pairs.

Here is a example of code to run knowledge distillation training using PyLate:

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

# Load the datasets required for knowledge distillation (train, queries, documents)
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

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Define the base model, training parameters, and output directory
model_name = "bert-base-uncased"
batch_size = 16
num_train_epochs = 1
output_dir = "output/distillation_run-bert-base"

# Initialize the ColBERT model
model = models.ColBERT(model_name_or_path=model_name)

# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=False,
    bf16=False,
    logging_steps=10,
    run_name="distillation_run-bert-base",
    learning_rate=1e-5,
)

# Use the Distillation loss function for training
train_loss = losses.Distillation(model=model)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

# Start the training process
trainer.train()
```

## Sentence Transformers Training Arguments

PyLate is built on top of SentenceTransformer, you can thus use the same arguments you already are familiar with to control the training. 
The table below lists the arguments for the `SentenceTransformerTrainingArguments` class. Feel free to refer to the [SentenceTransformers](https://sbert.net/docs/sentence_transformer/training_overview.html#) library documentation for more information

=== "Table"
| Parameter                         | Name                                 | Definition                                                                                                                                                                                                                                                                     | Training Performance |  Observing Performance |
|------------------------------------|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|
| `output_dir`                       | `str`                                | The output directory where the model predictions and checkpoints will be written.                                                                                                                                                                                               |                                                           |                                                            |
| `overwrite_output_dir`             | `bool`, *optional*, defaults to `False`| If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory.                                                                                                                                      |                                                           |                                                            |
| `do_train`                         | `bool`, *optional*, defaults to `False`| Whether to run training or not. Intended to be used by your training/evaluation scripts.                                                                                                                                                                                        |                                                           |                                                            |
| `do_eval`                          | `bool`, *optional*                   | Whether to run evaluation on the validation set. Will be `True` if `eval_strategy` is not `"no"`. Intended to be used by your training/evaluation scripts.                                                                                                                      |                                                           |                                                            |
| `do_predict`                       | `bool`, *optional*, defaults to `False`| Whether to run predictions on the test set or not. Intended to be used by your training/evaluation scripts.                                                                                                                                                                      |                                                           |                                                            |
| `eval_strategy`                    | `str` or `~trainer_utils.IntervalStrategy`, *optional*, defaults to `"no"`| The evaluation strategy to adopt during training. Possible values are `"no"`, `"steps"`, or `"epoch"`.                                                                                                                                                                         |                                                           | ✅                                                         |
| `prediction_loss_only`             | `bool`, *optional*, defaults to `False`| When performing evaluation and generating predictions, only returns the loss.                                                                                                                                                                                                   |                                                           |                                                            |
| `per_device_train_batch_size`      | `int`, *optional*, defaults to 8      | The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.                                                                                                                                                                                                                   | ✅                                                         |                                                            |
| `per_device_eval_batch_size`       | `int`, *optional*, defaults to 8      | The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.                                                                                                                                                                                                                 | ✅                                                         |                                                            |
| `gradient_accumulation_steps`      | `int`, *optional*, defaults to 1      | Number of updates steps to accumulate gradients before performing a backward/update pass.                                                                                                                                                                                       | ✅                                                         |                                                            |
| `eval_accumulation_steps`          | `int`, *optional*                    | Number of predictions steps to accumulate the output tensors before moving the results to CPU.                                                                                                                                                                                  | ✅                                                         |                                                            |
| `eval_delay`                       | `float`, *optional*                  | Number of epochs or steps to wait before the first evaluation depending on `eval_strategy`.                                                                                                                                                                                     |                                                           |                                                            |
| `torch_empty_cache_steps`          | `int`, *optional*                    | Number of steps to wait before calling `torch.<device>.empty_cache()` to avoid CUDA out-of-memory errors.                                                                                                                                                                       |                                                           |                                                            |
| `learning_rate`                    | `float`, *optional*, defaults to 5e-5| The initial learning rate for `AdamW` optimizer.                                                                                                                                                                                                                                | ✅                                                         |                                                            |                                                                                                                                                    |                                                           |                                                            |
| `num_train_epochs`                 | `float`, *optional*, defaults to 3.0 | Total number of training epochs to perform.                                                                                                                                                                                                                                     | ✅                                                         |                                                            |
| `max_steps`                        | `int`, *optional*, defaults to -1     | If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.                                                                                                                                                                       | ✅                                                         |                                                            |
| `lr_scheduler_type`                | `str` or `SchedulerType`, *optional*, defaults to `"linear"`| The scheduler type to use.                                                                                                                                                                                                                                                     | ✅                                                         |                                                            |
| `lr_scheduler_kwargs`              | `dict`, *optional*, defaults to {}    | Extra arguments for the learning rate scheduler.                                                                                                                                                                                                                                |                                                           |                                                            |
| `warmup_ratio`                     | `float`, *optional*, defaults to 0.0 | Ratio of total training steps used for linear warmup from 0 to `learning_rate`.                                                                                                                                                                                                 | ✅                                                         |                                                            |
| `warmup_steps`                     | `int`, *optional*, defaults to 0      | Number of steps used for linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.                                                                                                                                                                       |                                                           |                                                            |
| `log_level`                        | `str`, *optional*, defaults to `passive`| Logger log level to use on the main process.                                                                                                                                                                                                                                    |                                                           | ✅                                                         |
| `log_level_replica`                | `str`, *optional*, defaults to `"warning"`| Logger log level to use on replicas. Same choices as `log_level`.                                                                                                                                                                                                               |                                                           |                                                            |
| `log_on_each_node`                 | `bool`, *optional*, defaults to `True`| Whether to log using `log_level` once per node or only on the main node.                                                                                                                                                                                                        |                                                           |                                                            |
| `logging_dir`                      | `str`, *optional*                    | TensorBoard log directory.                                                                                                                                                                                                                                                     |                                                           |                                                            |
| `logging_strategy`                 | `str` or `~trainer_utils.IntervalStrategy`, *optional*, defaults to `"steps"`| The logging strategy to adopt during training. Possible values are `"no"`, `"epoch"`, or `"steps"`.                                                                                                                                                                            |                                                           | ✅                                                         |
| `logging_first_step`               | `bool`, *optional*, defaults to `False`| Whether to log the first `global_step` or not.                                                                                                                                                                                                                                  |                                                           |                                                            |
| `logging_steps`                    | `int` or `float`, *optional*, defaults to 500| Number of update steps between two logs if `logging_strategy="steps"`.                                                                                                                                                                                                          |                                                           | ✅                                                         |
| `logging_nan_inf_filter`           | `bool`, *optional*, defaults to `True`| Whether to filter `nan` and `inf` losses for logging.                                                                                                                                                                                                                           |                                                           |                                                            |
| `save_strategy`                    | `str` or `~trainer_utils.IntervalStrategy`, *optional*, defaults to `"steps"`| The checkpoint save strategy to adopt during training.                                                                                                                                                                                                                          |                                                           | ✅                                                         |
| `save_steps`                       | `int` or `float`, *optional*, defaults to 500| Number of update steps before two checkpoint saves if `save_strategy="steps"`.                                                                                                                                                                                                  |                                                           | ✅                                                         |
| `save_total_limit`                 | `int`, *optional*                    | Limit for total number of checkpoints.                                                                                                                                                                                                                                          |                                                           | ✅                                                         |
| `save_safetensors`                 | `bool`, *optional*, defaults to `True`| Use safetensors saving and loading for state dicts instead of default `torch.load` and `torch.save`.                                                                                                                                                                            |                                                           |                                                            |
| `save_on_each_node`                | `bool`, *optional*, defaults to `False`| Whether to save models and checkpoints on each node or only on the main one during multi-node distributed training.                                                                                                                                                              |                                                           |                                                            |
| `seed`                             | `int`, *optional*, defaults to 42     | Random seed set at the beginning of training for reproducibility.                                                                                                                                                                                                               |                                                           |                                                            |
| `auto_find_batch_size`             | `bool`, *optional*, defaults to `False`| Whether to find a batch size that will fit into memory automatically.                                                                                                                                                                                                           | ✅                                                         |                                                            |
| `fp16`                             | `bool`, *optional*, defaults to `False`| Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.                                                                                                                                                                                               | ✅                                                         |                                                            |
| `bf16`                             | `bool`, *optional*, defaults to `False`| Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training.                                                                                                                                                                                               | ✅                                                         |                                                            |
| `push_to_hub`                      | `bool`, *optional*, defaults to `False`| Whether to push the model to the Hub every time the model is saved.                                                                                                                                                                                                             |                                                           | ✅                                                         |
| `hub_model_id`                     | `str`, *optional*                    | The name of the repository to keep in sync with the local `output_dir`.                                                                                                                                                                                                         |                                                           | ✅                                                         |
| `hub_strategy`                     | `str` or `~trainer_utils.HubStrategy`, *optional*, defaults to `"every_save"`| Defines the scope of what is pushed to the Hub and when.                                                                                                                                                                                                                        |                                                           | ✅                                                         |
| `hub_private_repo`                 | `bool`, *optional*, defaults to `False`| If `True`, the Hub repo will be set to private.                                                                                                                                                                                                                                 |                                                           | ✅                                                         |
| `load_best_model_at_end`           | `bool`, *optional*, defaults to `False`| Whether or not to load the best model found during training at the end of training.                                                                                                                                                                                             |                                                           | ✅                                                         |
| `report_to`                        | `str` or `List[str]`, *optional*, defaults to `"all"`| The list of integrations to report the results and logs to.                                                                                                                                                                                                                     |                                                           | ✅                                                         |



## Sentence Transformer Trainer arguments

=== "Table"

    | Parameter   | Name                                                                                             | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    |-------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | model       | `~sentence_transformers.SentenceTransformer`, *optional*                                          | The model to train, evaluate, or use for predictions. If not provided, a `model_init` must be passed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
    | args        | `~sentence_transformers.training_args.SentenceTransformerTrainingArguments`, *optional*           | The arguments to tweak for training. Defaults to a basic instance of `SentenceTransformerTrainingArguments` with the `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.                                                                                                                                                                                                                                                                                                                                                                      |
    | train_dataset | `datasets.Dataset`, `datasets.DatasetDict`, or `Dict[str, datasets.Dataset]`, *optional*        | The dataset to use for training. Must have a format accepted by your loss function. Refer to `Training Overview > Dataset Format`.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | eval_dataset | `datasets.Dataset`, `datasets.DatasetDict`, or `Dict[str, datasets.Dataset]`, *optional*         | The dataset to use for evaluation. Must have a format accepted by your loss function. Refer to `Training Overview > Dataset Format`.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    | loss        | `torch.nn.Module`, `Dict[str, torch.nn.Module]`, Callable, or Dict[str, Callable], *optional*     | The loss function to use for training. It can be a loss class instance, a dictionary mapping dataset names to loss instances, a function returning a loss instance given a model, or a dictionary mapping dataset names to such functions. Defaults to `CoSENTLoss` if not provided.                                                                                                                                                                                                                                                                                                    |
    | evaluator   | `~sentence_transformers.evaluation.SentenceEvaluator` or `List[~sentence_transformers.evaluation.SentenceEvaluator]`, *optional* | The evaluator instance for useful metrics during training. Can be used with or without an `eval_dataset`. A list of evaluators will be wrapped in a `SequentialEvaluator` to run sequentially. Generally, evaluator metrics are more useful than loss values from `eval_dataset`.                                                                                                                                                                                                                                                                                                       |
    | callbacks   | `List[transformers.TrainerCallback]`, *optional*                                                  | A list of callbacks to customize the training loop. Adds to the list of default callbacks. To remove a default callback, use the `Trainer.remove_callback` method.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
    | optimizers  | `Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)` | A tuple containing the optimizer and scheduler to use. Defaults to an instance of `torch.optim.AdamW` for the model and a scheduler given by `transformers.get_linear_schedule_with_warmup`, controlled by `args`.                                                                                                                                                                                                                                                                                                                                                                      |


