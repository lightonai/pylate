# Available models

!!! tip
    Following an update, all the models trained using the stanford-nlp ColBERT library or RAGatouille should be compatible with PyLate natively (including their configurations).
    You can simply load the model in PyLate:

    ```python
    from pylate import models

    model = models.ColBERT(
        model_name_or_path="colbert-ir/colbertv2.0",
    )
    ```
    or
    ```python
    model = models.ColBERT(
        model_name_or_path="jinaai/jina-colbert-v2",
        trust_remote_code=True,
    )
    ```


Here is a list of some of the pre-trained ColBERT models available in PyLate along with their results on BEIR:

=== "Table"

| Model                                 | BEIR AVG | NFCorpus | SciFact | SCIDOCS | FiQA2018 | TRECCOVID | HotpotQA | Touche2020 | ArguAna | ClimateFEVER | FEVER | QuoraRetrieval | NQ   | DBPedia |
|---------------------------------------|----------|----------|---------|---------|----------|-----------|----------|------------|---------|--------------|-------|----------------|------|---------|
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0)                 | 50.02    | 33.8     | 69.3    | 15.4    | 35.6     | 73.3      | 66.7     | 26.3       | 46.3    | 17.6         | 78.5  | 85.2           | 56.2 | 44.6    |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 53.79    | 37.3     | 74.77   | 18.42   | 41.15    | 84.59     | 76.11    | 25.69      | 50.09   | 33.07        | 90.96 | 87.72          | 59.1 | 45.58   |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 53.1    | 34.6     | 67.8   | 18.6   | 40.8    | 83.4     | 76.6    | 27.4      | 36.6   | 23.9        | 80.05 | 88.7          | 64.0 | 47.1   |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 54.89    | 37.93     | 76.34   | 19.06   | 48.51    | 83.59     | 77.32    | 31.23      | 48.51   | 30.62       | 87.44 | 86.61          | 61.8 | 48.3   |



???+ note
    `lightonai/colbertv2.0` is the original [ColBERTv2 model](https://huggingface.co/colbert-ir/colbertv2.0/tree/main) made compatible with PyLate before we supported loading directly model from Stanford-NLP. We thank Omar Khattab for allowing us to share the model on PyLate.


# Defining dense layers
By default, if you use a base model to create a PyLate model, it'll add a dense layer projecting the output dimension of the model to `embedding_size`. If you did not specify any `embedding_size`, it'll default to 128.

```python
model = models.ColBERT("bert-base-uncased")
```

If you create a PyLate model from a sentence-transformers model, it'll load the dense layer of this model and only add another one **if you specified an embedding_size and it is not matching the size of the last dense layer of the ST model.

If you do not want to use the dense layers of the ST model (but still want to use its base weights), you should use the modular syntax:
```python

import torch
from sentence_transformers.models import Transformer
from pylate import models

base_model = Transformer("answerdotai/ModernBERT-base")

dense_1 = models.Dense(
    in_features=768,
    out_features=512,
    bias=False,
    activation_function=torch.nn.GELU(),
)
dense_2 = models.Dense(
    in_features=512,
    out_features=128,
    bias=False,
    activation_function=torch.nn.Identity(),
)

model = models.ColBERT(
    modules=[base_model, dense_1, dense_2],
    document_length=300,
    query_length=32,
)

ColBERT(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False, 'architecture': 'ModernBertModel'})
  (1): Dense({'in_features': 768, 'out_features': 512, 'bias': False, 'activation_function': 'torch.nn.modules.activation.GELU', 'use_residual': False})
  (2): Dense({'in_features': 512, 'out_features': 128, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)
```

It also allows you to define the activation function and use multiple dense.
Please note that you can also _append_ layers to existing models as well as remove them, so you can really create the modules you want
```python
import torch
from pylate import models
model = models.ColBERT("google/embeddinggemma-300m")
ColBERT(
  (0): Transformer({'max_seq_length': 2048, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
  (1): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
  (2): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)

dense_1 = models.Dense(
    in_features=768,
    out_features=128,
    bias=False,
    activation_function=torch.nn.Identity(),
    use_residual=False,
)

model.append(dense_1)
ColBERT(
  (0): Transformer({'max_seq_length': 2048, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
  (1): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
  (2): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
  (3): Dense({'in_features': 768, 'out_features': 128, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)

del model[3]
ColBERT(
  (0): Transformer({'max_seq_length': 2048, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
  (1): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
  (2): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)
```


!!! tip
    [MixedBread study](https://arxiv.org/abs/2510.12327) showed that it is beneficial to use MLPs to do the projection rather than a simple dense layer. The study explore different depths, activation functions and the use of residual layers. Please check the paper for a more thorough analysis.
    ```python
    import torch
    from sentence_transformers.models import Transformer
    from pylate import models

    base_model = Transformer("jhu-clsp/ettin-encoder-32m")

    dense_1 = models.Dense(
        in_features=384,
        out_features=768,
        bias=False,
        activation_function=torch.nn.Identity(),
        use_residual=True,
    )
    dense_2 = models.Dense(
        in_features=768,
        out_features=384,
        bias=False,
        activation_function=torch.nn.Identity(),
        use_residual=False,
    )

    model = models.ColBERT(
        modules=[base_model, dense_1, dense_2],
    )
    ```
