---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name if model_name else "Sentence Transformer model" }}

This is a [PyLate](https://github.com/lightonai/pylate) model{% if base_model %} finetuned from [{{ base_model }}](https://huggingface.co/{{ base_model }}){% else %} trained{% endif %}{% if train_datasets | selectattr("name") | list %} on the {% for dataset in (train_datasets | selectattr("name")) %}{% if dataset.id %}[{{ dataset.name if dataset.name else dataset.id }}](https://huggingface.co/datasets/{{ dataset.id }}){% else %}{{ dataset.name }}{% endif %}{% if not loop.last %}{% if loop.index == (train_datasets | selectattr("name") | list | length - 1) %} and {% else %}, {% endif %}{% endif %}{% endfor %} dataset{{"s" if train_datasets | selectattr("name") | list | length > 1 else ""}}{% endif %}. It maps sentences & paragraphs to sequences of {{ output_dimensionality }}-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
{% if base_model -%}
    {%- if base_model_revision -%}
    - **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }}) <!-- at revision {{ base_model_revision }} -->
    {%- else -%}
    - **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }})
    {%- endif -%}
{%- else -%}
    <!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
{%- endif %}
- **Document Length:** {{ document_length }} tokens
- **Query Length:** {{ query_length }} tokens
- **Output Dimensionality:** {{ output_dimensionality }} tokens
- **Similarity Function:** MaxSim
{% if train_datasets | selectattr("name") | list -%}
    - **Training Dataset{{"s" if train_datasets | selectattr("name") | list | length > 1 else ""}}:**
    {%- for dataset in (train_datasets | selectattr("name")) %}
        {%- if dataset.id %}
    - [{{ dataset.name if dataset.name else dataset.id }}](https://huggingface.co/datasets/{{ dataset.id }})
        {%- else %}
    - {{ dataset.name }}
        {%- endif %}
    {%- endfor %}
{%- else -%}
    <!-- - **Training Dataset:** Unknown -->
{%- endif %}
{% if language -%}
    - **Language{{"s" if language is not string and language | length > 1 else ""}}:**
    {%- if language is string %} {{ language }}
    {%- else %} {% for lang in language -%}
            {{ lang }}{{ ", " if not loop.last else "" }}
        {%- endfor %}
    {%- endif %}
{%- else -%}
    <!-- - **Language:** Unknown -->
{%- endif %}
{% if license -%}
    - **License:** {{ license }}
{%- else -%}
    <!-- - **License:** Unknown -->
{%- endif %}

### Model Sources

- **Documentation:** [PyLate Documentation](https://lightonai.github.io/pylate/)
- **Repository:** [PyLate on GitHub](https://github.com/lightonai/pylate)
- **Hugging Face:** [PyLate models on Hugging Face](https://huggingface.co/models?library=PyLate)

### Full Model Architecture

```
{{ model_string }}
```

## Usage
First install the PyLate library:

```bash
pip install -U pylate
```

### Retrieval 

PyLate provides a streamlined interface to index and retrieve documents using ColBERT models. The index leverages the Voyager HNSW index to efficiently handle document embeddings and enable fast retrieval.

#### Indexing documents

First, load the ColBERT model and initialize the Voyager index, then encode and index your documents:

```python
from pylate import indexes, models, retrieve

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path={{ model_id | default('sentence_transformers_model_id', true) }},
)

# Step 2: Initialize the Voyager index
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
    override=True,  # This overwrites the existing index if any
)

# Step 3: Encode the documents
documents_ids = ["1", "2", "3"]
documents = ["document 1 text", "document 2 text", "document 3 text"]

documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
    show_progress_bar=True,
)

# Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

Note that you do not have to recreate the index and encode the documents every time. Once you have created an index and added the documents, you can re-use the index later by loading it:

```python
# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
)
```

#### Retrieving top-k documents for queries

Once the documents are indexed, you can retrieve the top-k most relevant documents for a given set of queries.
To do so, initialize the ColBERT retriever with the index you want to search in, encode the queries and then retrieve the top-k documents to get the top matches ids and relevance scores:

```python
# Step 1: Initialize the ColBERT retriever
retriever = retrieve.ColBERT(index=index)

# Step 2: Encode the queries
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
    show_progress_bar=True,
)

# Step 3: Retrieve top-k documents
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings, 
    k=10,  # Retrieve the top 10 matches for each query
)
```

### Reranking
If you only want to use the ColBERT model to perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use rank function and pass the queries and documents to rerank:

```python
from pylate import rank, models

queries = [
    "query A",
    "query B",
]

documents = [
    ["document A", "document B"],
    ["document 1", "document C", "document B"],
]

documents_ids = [
    [1, 2],
    [1, 3, 2],
]

model = models.ColBERT(
    model_name_or_path={{ model_id | default('sentence_transformers_model_id', true) }},
)

queries_embeddings = model.encode(
    queries,
    is_query=True,
)

documents_embeddings = model.encode(
    documents,
    is_query=False,
)

reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->
{% if eval_metrics %}
## Evaluation

### Metrics
{% for metrics in eval_metrics %}
#### {{ metrics.description }}
{% if metrics.dataset_name %}* Dataset: `{{ metrics.dataset_name }}`{% endif %}
* Evaluated with {% if metrics.class_name.startswith("sentence_transformers.") %}[<code>{{ metrics.class_name.split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.{{ metrics.class_name.split(".")[-1] }}){% else %}<code>{{ metrics.class_name }}</code>{% endif %}

{{ metrics.table }}
{%- endfor %}{% endif %}
<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details
{% for dataset_type, dataset_list in [("training", train_datasets), ("evaluation", eval_datasets)] %}{% if dataset_list %}
### {{ dataset_type.title() }} Dataset{{"s" if dataset_list | length > 1 else ""}}
{% for dataset in dataset_list %}
#### {{ dataset['name'] or 'Unnamed Dataset' }}

{% if dataset['name'] %}* Dataset: {% if 'id' in dataset %}[{{ dataset['name'] }}](https://huggingface.co/datasets/{{ dataset['id'] }}){% else %}{{ dataset['name'] }}{% endif %}
{%- if 'revision' in dataset and 'id' in dataset %} at [{{ dataset['revision'][:7] }}](https://huggingface.co/datasets/{{ dataset['id'] }}/tree/{{ dataset['revision'] }}){% endif %}{% endif %}
{% if dataset['size'] %}* Size: {{ "{:,}".format(dataset['size']) }} {{ dataset_type }} samples
{% endif %}* Columns: {% if dataset['columns'] | length == 1 %}{{ dataset['columns'][0] }}{% elif dataset['columns'] | length == 2 %}{{ dataset['columns'][0] }} and {{ dataset['columns'][1] }}{% else %}{{ dataset['columns'][:-1] | join(', ') }}, and {{ dataset['columns'][-1] }}{% endif %}
{% if dataset['stats_table'] %}* Approximate statistics based on the first {{ [dataset['size'], 1000] | min }} samples:
{{ dataset['stats_table'] }}{% endif %}{% if dataset['examples_table'] %}* Samples:
{{ dataset['examples_table'] }}{% endif %}* Loss: {% if dataset["loss"]["fullname"].startswith("sentence_transformers.") %}[<code>{{ dataset["loss"]["fullname"].split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#{{ dataset["loss"]["fullname"].split(".")[-1].lower() }}){% else %}<code>{{ dataset["loss"]["fullname"] }}</code>{% endif %}{% if "config_code" in dataset["loss"] %} with these parameters:
{{ dataset["loss"]["config_code"] }}{% endif %}
{% endfor %}{% endif %}{% endfor -%}

{% if all_hyperparameters %}
### Training Hyperparameters
{% if non_default_hyperparameters -%}
#### Non-Default Hyperparameters

{% for name, value in non_default_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}{%- endif %}
#### All Hyperparameters
<details><summary>Click to expand</summary>

{% for name, value in all_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}
</details>
{% endif %}

{%- if eval_lines %}
### Training Logs
{% if hide_eval_lines %}<details><summary>Click to expand</summary>

{% endif -%}
{{ eval_lines }}{% if explain_bold_in_eval %}
* The bold row denotes the saved checkpoint.{% endif %}
{%- if hide_eval_lines %}
</details>{% endif %}
{% endif %}

{%- if co2_eq_emissions %}
### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Energy Consumed**: {{ "%.3f"|format(co2_eq_emissions["energy_consumed"]) }} kWh
- **Carbon Emitted**: {{ "%.3f"|format(co2_eq_emissions["emissions"] / 1000) }} kg of CO2
- **Hours Used**: {{ co2_eq_emissions["hours_used"] }} hours

### Training Hardware
- **On Cloud**: {{ "Yes" if co2_eq_emissions["on_cloud"] else "No" }}
- **GPU Model**: {{ co2_eq_emissions["hardware_used"] or "No GPU used" }}
- **CPU Model**: {{ co2_eq_emissions["cpu_model"] }}
- **RAM Size**: {{ "%.2f"|format(co2_eq_emissions["ram_total_size"]) }} GB
{% endif %}
### Framework Versions
- Python: {{ version["python"] }}
- Sentence Transformers: {{ version["sentence_transformers"] }}
- PyLate: {{ version["pylate"] }}
- Transformers: {{ version["transformers"] }}
- PyTorch: {{ version["torch"] }}
- Accelerate: {{ version["accelerate"] }}
- Datasets: {{ version["datasets"] }}
- Tokenizers: {{ version["tokenizers"] }}


## Citation

### BibTeX
{% for loss_name, citation in citations.items() %}
#### {{ loss_name }}
```bibtex
{{ citation | trim }}
```
{% endfor %}
<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->