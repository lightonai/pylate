# ColBERT Training
PyLate training is based on Sentence Transformer (and thus transformers) trainer, enabling a lot of functionality such multi-GPU and FP16/BF16 training as well as logging to Weights & Biases out-of-the-box. This allows efficient, and scalable training.

???+ info
    There are two primary ways to train ColBERT models using PyLate:

    1. **Contrastive Loss**: Simplest method, it only requires a dataset containing triplets, each consisting of a query, a positive document (relevant to the query), and a negative document (irrelevant to the query). This method trains the model to maximize the similarity between the query and the positive document, while minimizing it with the negative document.

    2. **Knowledge Distillation**: To train a ColBERT model using knowledge distillation, you need to provide a dataset with three components: queries, documents, and the relevance scores between them. This method compresses the knowledge of a larger model / more accurate model (cross-encoder) into a smaller one, using the relevance scores to guide the training process.

## Contrastive Training

ColBERT was originally trained using contrastive learning. This approach involves teaching the model to distinguish between relevant (positive) and irrelevant (negative) documents for a given query. The model is trained to maximize the similarity between a query and its corresponding positive document while minimizing the similarity with irrelevant documents.

PyLate uses contrastive learning with a triplet dataset, where each query is paired with one positive and one negative example. **This makes it fully compatible with any triplet datasets from the sentence-transformers library**.

During training, the model is optimized to maximize the similarity between the query and its positive example while minimizing the similarity with all negative examples and the positives from other queries in the batch. This approach leverages in-batch negatives for more effective learning.

Here is an example of code to run contrastive training with PyLate:

```python
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

# Define model parameters for contrastive training
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 32  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory
run_name = "contrastive-bert-base-uncased"
output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model makes the training faster
model = torch.compile(model)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define the loss function
train_loss = losses.Contrastive(model=model)

# Initialize the evaluator
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
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    learning_rate=3e-6,
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
Please note that temperature parameter has a [very high importance in contrastive learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf). A low temperature allows to focus more on the hardest elements in the batch, creating more discriminative representations but is more sensible to false negative. A temperature around 0.02 is often used in the literature:
```python
train_loss = losses.Contrastive(model=model, temperature=0.02)
```

As contrastive learning is not compatible with gradient accumulation, you can leverage [GradCache](https://arxiv.org/abs/2101.06983) to emulate bigger batch sizes without requiring more memory by using the `CachedContrastiveLoss` to define a mini_batch_size while increasing the `per_device_train_batch_size`:
```python
train_loss = losses.CachedContrastive(
        model=model, mini_batch_size=mini_batch_size
)
```
Finally, if you are in a multi-GPU setting, you can gather all the elements from the different GPUs to create even bigger batch sizes by setting `gather_across_devices` to `True` (for both `Contrastive` and `CachedContrastive` losses):
```python
train_loss = losses.Contrastive(model=model, gather_across_devices=True)
```

???+ tip
    Please note that for multi-GPU training, running ``python training.py`` **will use Data Parallel (DP) by default**. We strongly suggest using using Distributed Data Parallelism (DDP) using accelerate or torchrun: ``accelerate launch --num_processes num_gpu training.py``.

    Refer to this [documentation](https://sbert.net/docs/sentence_transformer/training/distributed.html) for more information.

???+ tip
    PyLate now features [NanoBEIREvaluator](https://x.com/tomaarsen/status/1857434642569138243), an evaluator that allows to run small versions of the BEIR datasets to get an idea of the performance on BEIR without taking too long to run.

    To use NanoBEIREvaluator, you can simply use ``evaluator=evaluation.NanoBEIREvaluator()`` as an argument of the ``SentenceTransformerTrainer``. You can select to run only a subset of the evaluations by specifying ``dataset_names``, e.g, ``evaluation.NanoBEIREvaluator(dataset_names=["SciFact", NFCorpus])``

## Knowledge Distillation Training

Training late-interaction models, such as ColBERT, has been shown to benefit from knowledge distillation compared to simpler contrastive learning approaches. Knowledge distillation training focuses on teaching ColBERT models to replicate the outputs of a more capable teacher model (e.g., a cross-encoder). This is achieved using a dataset that includes queries, documents, and the scores assigned by the teacher model to each query/document pair.

Below is an example of code to run knowledge distillation training using PyLate:

```python
import torch
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
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 16
num_train_epochs = 1
# Set the run name for logging and output directory
run_name = "knowledge-distillation-bert-base"
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model to make the training faster
model = torch.compile(model)

# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,
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

???+ tip
    Please note that for multi-GPU training, running ``python training.py`` **will use Data Parallel (DP) by default**. We strongly suggest using using Distributed Data Parallelism (DDP) using accelerate or torchrun: ``accelerate launch --num_processes num_gpu training.py``.

    Refer to this [documentation](https://sbert.net/docs/sentence_transformer/training/distributed.html) for more information.

### NanoBEIR evaluator
If you are training an English retrieval model, you can use [NanoBEIR evaluator](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6), which allows to run small version of BEIR to get quick validation results.
```python
evaluator=evaluation.NanoBEIREvaluator(),
```
You can select a subset of all the sets to run by adding the dataset names: `evaluation.NanoBEIREvaluator(dataset_names=["SciFact"])`

## ColBERT parameters
All the parameters of the ColBERT modeling can be found [here](https://lightonai.github.io/pylate/api/models/ColBERT/#parameters). Important parameters to consider are:

???+ info
    - `model_name_or_path` the name of the base encoder model or PyLate model to init from.
    - `embedding_size` the output size of the projection layer. Large values give more capacity to the model but are heavier to store.
    - `query_prefix` and `document_prefix` represents the strings that will be prepended to query and document respectively.
    - `query_length` and `document_length` set the maximum size of queries and documents. Queries will be padded/truncated to the maximum length while documents are only truncated.
    - `attend_to_expansion_tokens` define whether the model will attend to the query expansion tokens (padding of queries) or if only the expansion tokens will attend to the other tokens. In the original ColBERT, the tokens **do not attend** to expansion tokens.
    - `skiplist_words` is list of words to skip from the documents scoring (note that these tokens are used for encoding and are only skipped during the scoring), the default is the list of string.punctuation as in the original ColBERT.


## Sentence Transformers Training Arguments

PyLate is built on top of SentenceTransformer, so you can use the same arguments you are already familiar with to control the training process. The table below lists the arguments available in the SentenceTransformerTrainingArguments class. For more details, please refer to the [SentenceTransformers documentation](https://sbert.net/docs/sentence_transformer/training_overview.html#).

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
