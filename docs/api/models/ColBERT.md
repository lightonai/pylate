# ColBERT

Loads or creates a ColBERT model that can be used to map sentences / text to multi-vectors embeddings.



## Parameters

- **model_name_or_path** (*'str | None'*) – defaults to `None`

    If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from the Hugging Face Hub with that name.

- **modules** (*'Optional[Iterable[nn.Module]]'*) – defaults to `None`

    A list of torch Modules that should be called sequentially, can be used to create custom SentenceTransformer models from scratch.

- **device** (*'str | None'*) – defaults to `None`

    Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.

- **prompts** (*'dict[str, str] | None'*) – defaults to `None`

    A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text. The prompt text will be prepended before any text to encode. For example: `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the titles in "}`.

- **default_prompt_name** (*'str | None'*) – defaults to `None`

    The name of the prompt that should be used by default. If not set, no prompt will be applied.

- **similarity_fn_name** (*'Optional[str | SimilarityFunction]'*) – defaults to `None`

    The name of the similarity function to use. Valid options are "cosine", "dot", "euclidean", and "manhattan". If not set, it is automatically set to "cosine" if `similarity` or `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.

- **cache_folder** (*'str | None'*) – defaults to `None`

    Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.

- **trust_remote_code** (*'bool'*) – defaults to `False`

    Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should only be set to True for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.

- **revision** (*'str | None'*) – defaults to `None`

    The specific model version to use. It can be a branch name, a tag name, or a commit id, for a stored model on Hugging Face.

- **local_files_only** (*'bool'*) – defaults to `False`

    Whether or not to only look at local files (i.e., do not try to download the model).

- **token** (*'bool | str | None'*) – defaults to `None`

    Hugging Face authentication token to download private models.

- **use_auth_token** (*'bool | str | None'*) – defaults to `None`

    Deprecated argument. Please use `token` instead.

- **truncate_dim** (*'int | None'*) – defaults to `None`

    The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is only applicable during inference when :meth:`SentenceTransformer.encode` is called.

- **embedding_size** (*'int | None'*) – defaults to `None`

    The output size of the projection layer. Default to 128.

- **bias** (*'bool'*) – defaults to `False`

- **query_prefix** (*'str | None'*) – defaults to `None`

    Prefix to add to the queries.

- **document_prefix** (*'str | None'*) – defaults to `None`

    Prefix to add to the documents.

- **add_special_tokens** (*'bool'*) – defaults to `True`

    Add the prefix to the inputs.

- **truncation** (*'bool'*) – defaults to `True`

    Truncate the inputs to the encoder max lengths or use sliding window encoding.

- **query_length** (*'int | None'*) – defaults to `None`

    The length of the query to truncate/pad to with mask tokens. If set, will override the config value. Default to 32.

- **document_length** (*'int | None'*) – defaults to `None`

    The max length of the document to truncate. If set, will override the config value. Default to 180.

- **do_query_expansion** (*'bool | None'*) – defaults to `None`

    Whether to do query expansion. If True, will pad the query to the `query_length` with mask tokens. Default to True.

- **attend_to_expansion_tokens** (*'bool | None'*) – defaults to `None`

    Whether to attend to the expansion tokens in the attention layers model. If False, the original tokens will not only attend to the expansion tokens, only the expansion tokens will attend to the original tokens. Default is False (as in the original ColBERT codebase).

- **skiplist_words** (*'list[str] | None'*) – defaults to `None`

    A list of words to skip from the documents scoring (note that these tokens are used for encoding and are only skipped during the scoring). Default is the list of string.punctuation.

- **model_kwargs** (*'dict | None'*) – defaults to `None`

    Additional model configuration parameters to be passed to the Huggingface Transformers model. Particularly useful options are:  - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`. The     different options are:          1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified ``dtype``,         ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will get         loaded in ``torch.float`` (fp32).          2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be attempted         to be used. If this entry isn't found then next check the ``dtype`` of the first weight in the         checkpoint that's of a floating point type and use that as ``dtype``. This will load the model using         the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how the         model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32. - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of     `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention     <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),     or `"flash_attention_2"` (using `Dao-AILab/flash-attention     <https://github.com/Dao-AILab/flash-attention>`_). By default, if available, SDPA will be used for     torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.  See the `PreTrainedModel.from_pretrained <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_ documentation for more details.

- **tokenizer_kwargs** (*'dict | None'*) – defaults to `None`

    Additional tokenizer configuration parameters to be passed to the Huggingface Transformers tokenizer. See the `AutoTokenizer.from_pretrained <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_ documentation for more details.

- **config_kwargs** (*'dict | None'*) – defaults to `None`

    Additional model configuration parameters to be passed to the Huggingface Transformers config. See the `AutoConfig.from_pretrained <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_ documentation for more details.

- **model_card_data** (*'PylateModelCardData | None'*) – defaults to `None`

    A model card data object that contains information about the model. This is used to generate a model card when saving the model. If not set, a default model card data object is created.


## Attributes

- **device**

    Get torch.device from module, assuming that the whole module has one device. In case there are no PyTorch parameters, fall back to CPU.

- **dtype**

- **max_seq_length**

    Returns the maximal input sequence length for the model. Longer inputs will be truncated.  Returns:     int: The maximal input sequence length.  Example:     ::          from sentence_transformers import SentenceTransformer          model = SentenceTransformer("all-mpnet-base-v2")         print(model.max_seq_length)         # => 384

- **similarity**

    Compute the similarity between two collections of embeddings. The output will be a matrix with the similarity scores between all embeddings from the first parameter and all embeddings from the second parameter. This differs from `similarity_pairwise` which computes the similarity between each pair of embeddings. This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.  Args:     embeddings1 (Union[Tensor, ndarray]): [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.     embeddings2 (Union[Tensor, ndarray]): [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.  Returns:     Tensor: A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.  Example:     ::          >>> model = SentenceTransformer("all-mpnet-base-v2")         >>> sentences = [         ...     "The weather is so nice!",         ...     "It's so sunny outside.",         ...     "He's driving to the movie theater.",         ...     "She's going to the cinema.",         ... ]         >>> embeddings = model.encode(sentences, normalize_embeddings=True)         >>> model.similarity(embeddings, embeddings)         tensor([[1.0000, 0.7235, 0.0290, 0.1309],                 [0.7235, 1.0000, 0.0613, 0.1129],                 [0.0290, 0.0613, 1.0000, 0.5027],                 [0.1309, 0.1129, 0.5027, 1.0000]])         >>> model.similarity_fn_name         "cosine"         >>> model.similarity_fn_name = "euclidean"         >>> model.similarity(embeddings, embeddings)         tensor([[-0.0000, -0.7437, -1.3935, -1.3184],                 [-0.7437, -0.0000, -1.3702, -1.3320],                 [-1.3935, -1.3702, -0.0000, -0.9973],                 [-1.3184, -1.3320, -0.9973, -0.0000]])

- **similarity_fn_name**

    Return the name of the similarity function used by :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise`.  Returns:     Optional[str]: The name of the similarity function. Can be None if not set, in which case it will         default to "cosine" when first called. Examples -------- >>> model = ColBERT("bert-base-uncased") >>> model.similarity_fn_name     'MaxSim'

- **similarity_pairwise**

    Compute the similarity between two collections of embeddings. The output will be a vector with the similarity scores between each pair of embeddings. This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.  Args:     embeddings1 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.     embeddings2 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.  Returns:     Tensor: A [num_embeddings]-shaped torch tensor with pairwise similarity scores.  Example:     ::          >>> model = SentenceTransformer("all-mpnet-base-v2")         >>> sentences = [         ...     "The weather is so nice!",         ...     "It's so sunny outside.",         ...     "He's driving to the movie theater.",         ...     "She's going to the cinema.",         ... ]         >>> embeddings = model.encode(sentences, normalize_embeddings=True)         >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])         tensor([0.7235, 0.5027])         >>> model.similarity_fn_name         "cosine"         >>> model.similarity_fn_name = "euclidean"         >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])         tensor([-0.7437, -0.9973])

- **tokenizer**

    Property to get the tokenizer that is used by this model

- **transformers_model**

    Property to get the underlying transformers PreTrainedModel instance, if it exists. Note that it's possible for a model to have multiple underlying transformers models, but this property will return the first one it finds in the module hierarchy.  Returns:     PreTrainedModel or None: The underlying transformers model or None if not found.  Example:     ::          from sentence_transformers import SentenceTransformer          model = SentenceTransformer("all-mpnet-base-v2")          # You can now access the underlying transformers model         transformers_model = model.transformers_model         print(type(transformers_model))         # => <class 'transformers.models.mpnet.modeling_mpnet.MPNetModel'>


## Examples

```python
>>> from pylate import models

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
...     device="cpu",
... )

>>> embeddings = model.encode("Hello, how are you?")
>>> assert isinstance(embeddings, np.ndarray)

>>> embeddings = model.encode([
...     "Hello, how are you?",
...     "How is the weather today?"
... ])

>>> assert len(embeddings) == 2
>>> assert isinstance(embeddings[0], np.ndarray)
>>> assert isinstance(embeddings[1], np.ndarray)

>>> embeddings = model.encode([
...     [
...         "Hello, how are you?",
...         "How is the weather today?"
...     ],
...     [
...         "Hello, how are you?",
...         "How is the weather today?"
...     ],
... ])

>>> assert len(embeddings) == 2

>>> model.save_pretrained("test-model")

>>> model = models.ColBERT("test-model")

>>> embeddings = model.encode([
...     "Hello, how are you?",
...     "How is the weather today?"
... ])

>>> assert len(embeddings) == 2
>>> assert isinstance(embeddings[0], np.ndarray)
>>> assert isinstance(embeddings[1], np.ndarray)
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **args**
    - **kwargs**

???- note "active_adapter"

???- note "active_adapters"

    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

    Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters for inference) returns the list of all active adapters so that users can deal with them accordingly.  For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return a single string.


???- note "add_adapter"

    Adds a fresh new adapter to the current model for training purposes. If no adapter name is passed, a default name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the default adapter name).

    Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.  Args:     *args:         Positional arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter     **kwargs:         Keyword arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter

    **Parameters**

    - **args**
    - **kwargs**

???- note "add_module"

    Add a child module to the current module.

    The module can be accessed as an attribute using the given name.  Args:     name (str): name of the child module. The child module can be         accessed from this module using the given name     module (Module): child module to be added to the module.

    **Parameters**

    - **name**     (*str*)
    - **module**     (*Optional[ForwardRef('Module')]*)

???- note "append"

    Append a given module to the end.

    Args:     module (nn.Module): module to append  Example::      >>> import torch.nn as nn     >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))     >>> n.append(nn.Linear(3, 4))     Sequential(         (0): Linear(in_features=1, out_features=2, bias=True)         (1): Linear(in_features=2, out_features=3, bias=True)         (2): Linear(in_features=3, out_features=4, bias=True)     )

    **Parameters**

    - **module**     (*'Module'*)

???- note "apply"

    Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

    Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).  Args:     fn (:class:`Module` -> None): function to be applied to each submodule  Returns:     Module: self  Example::      >>> @torch.no_grad()     >>> def init_weights(m):     >>>     print(m)     >>>     if type(m) == nn.Linear:     >>>         m.weight.fill_(1.0)     >>>         print(m.weight)     >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))     >>> net.apply(init_weights)     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[1., 1.],             [1., 1.]], requires_grad=True)     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[1., 1.],             [1., 1.]], requires_grad=True)     Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )

    **Parameters**

    - **fn**     (*Callable[[ForwardRef('Module')], NoneType]*)

???- note "bfloat16"

    Casts all floating point parameters and buffers to ``bfloat16`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "buffers"

    Return an iterator over module buffers.

    Args:     recurse (bool): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module.  Yields:     torch.Tensor: module buffer  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for buf in model.buffers():     >>>     print(type(buf), buf.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`

???- note "check_peft_compatible_model"

???- note "children"

    Return an iterator over immediate children modules.

    Yields:     Module: a child module


???- note "compile"

    Compile this Module's forward using :func:`torch.compile`.

    This Module's `__call__` method is compiled and all arguments are passed as-is to :func:`torch.compile`.  See :func:`torch.compile` for details on the arguments for this function.

    **Parameters**

    - **args**
    - **kwargs**

???- note "cpu"

    Move all model parameters and buffers to the CPU.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "cuda"

    Move all model parameters and buffers to the GPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on GPU while being optimized.  .. note::     This method modifies the module in-place.  Args:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[torch.device, int, NoneType]*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.

???- note "delete_adapter"

    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

    Delete an adapter's LoRA layers from the underlying model.  Args:     *args:         Positional arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter     **kwargs:         Keyword arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter

    **Parameters**

    - **args**
    - **kwargs**

???- note "disable_adapters"

    Disable all adapters that are attached to the model. This leads to inferring with the base model only.


???- note "double"

    Casts all floating point parameters and buffers to ``double`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "enable_adapters"

    Enable adapters that are attached to the model. The model will use `self.active_adapter()`


???- note "encode"

    Computes sentence embeddings.

    **Parameters**

    - **sentences**     (*'str | list[str]'*)
    - **prompt_name**     (*'str | None'*)     – defaults to `None`
    - **prompt**     (*'str | None'*)     – defaults to `None`
    - **batch_size**     (*'int'*)     – defaults to `32`
    - **show_progress_bar**     (*'bool'*)     – defaults to `None`
    - **precision**     (*"Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']"*)     – defaults to `float32`
    - **convert_to_numpy**     (*'bool'*)     – defaults to `True`
    - **convert_to_tensor**     (*'bool'*)     – defaults to `False`
    - **padding**     (*'bool'*)     – defaults to `False`
    - **device**     (*'str'*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.
    - **normalize_embeddings**     (*'bool'*)     – defaults to `True`
    - **is_query**     (*'bool'*)     – defaults to `True`
    - **pool_factor**     (*'int'*)     – defaults to `1`
    - **protected_tokens**     (*'int'*)     – defaults to `1`

???- note "encode_document"

    Computes sentence embeddings specifically optimized for document/passage representation.

    This method is a specialized version of :meth:`encode` that differs in exactly two ways:  1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "document" prompt,    if available in the model's ``prompts`` dictionary. 2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.models.Router`    module, it will use the "document" task type to route the input through the appropriate submodules.  .. tip::      If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,     your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks     with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.      Note that :meth:`encode` is the most general method and can be used for any task, including Information     Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three     methods will return identical embeddings.  Args:     sentences (Union[str, List[str]]): The sentences to embed.     prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,         which is either set in the constructor or loaded from the model configuration. For example if         ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What         is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence         is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.     prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the         sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"         because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.     batch_size (int, optional): The batch size used for the computation. Defaults to 32.     show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.     output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:         "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,         to get all output values. Defaults to "sentence_embedding".     precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.         Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.         Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for         reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".     convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.         Defaults to True.     convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.         Defaults to False.     device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:          - A single device string (e.g., "cuda:0", "cpu") for single-process encoding         - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute           encoding across multiple processes         - None to auto-detect available device for single-process encoding         If a list is provided, multi-process encoding will be used. Defaults to None.     normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,         the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.     truncate_dim (int, optional): The dimension to truncate sentence embeddings to.         Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,         i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.         Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference         is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``         from the model initialization is used. Defaults to None.     pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`         for multi-process encoding. If provided, the encoding will be distributed across multiple processes.         This is recommended for large datasets and when multiple GPUs are available. Defaults to None.     chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when         ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.  Returns:     Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.     If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,     a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.  Example:     ::          from sentence_transformers import SentenceTransformer          # Load a pre-trained SentenceTransformer model         model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")          # Encode some documents         documents = [             "This research paper discusses the effects of climate change on marine life.",             "The article explores the history of artificial intelligence development.",             "This document contains technical specifications for the new product line.",         ]          # Using document-specific encoding         embeddings = model.encode_document(documents)         print(embeddings.shape)         # (3, 768)

    **Parameters**

    - **sentences**     (*'str | list[str] | np.ndarray'*)
    - **prompt_name**     (*'str | None'*)     – defaults to `None`
    - **prompt**     (*'str | None'*)     – defaults to `None`
    - **batch_size**     (*'int'*)     – defaults to `32`
    - **show_progress_bar**     (*'bool | None'*)     – defaults to `None`
    - **output_value**     (*"Literal['sentence_embedding', 'token_embeddings'] | None"*)     – defaults to `sentence_embedding`
    - **precision**     (*"Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']"*)     – defaults to `float32`
    - **convert_to_numpy**     (*'bool'*)     – defaults to `True`
    - **convert_to_tensor**     (*'bool'*)     – defaults to `False`
    - **device**     (*'str | list[str | torch.device] | None'*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.
    - **normalize_embeddings**     (*'bool'*)     – defaults to `False`
    - **truncate_dim**     (*'int | None'*)     – defaults to `None`
        The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is only applicable during inference when :meth:`SentenceTransformer.encode` is called.
    - **pool**     (*"dict[Literal['input', 'output', 'processes'], Any] | None"*)     – defaults to `None`
    - **chunk_size**     (*'int | None'*)     – defaults to `None`
    - **kwargs**

???- note "encode_multi_process"

    Encodes a list of sentences using multiple processes and GPUs via :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`. The sentences are chunked into smaller packages and sent to individual processes, which encode them on different GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

    **Parameters**

    - **sentences**     (*'list[str]'*)
    - **pool**     (*'dict[str, object]'*)
    - **prompt_name**     (*'str | None'*)     – defaults to `None`
    - **prompt**     (*'str | None'*)     – defaults to `None`
    - **batch_size**     (*'int'*)     – defaults to `32`
    - **chunk_size**     (*'int'*)     – defaults to `None`
    - **precision**     (*"Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']"*)     – defaults to `float32`
    - **normalize_embeddings**     (*'bool'*)     – defaults to `True`
    - **padding**     (*'bool'*)     – defaults to `False`
    - **is_query**     (*'bool'*)     – defaults to `True`
    - **pool_factor**     (*'int'*)     – defaults to `1`
    - **protected_tokens**     (*'int'*)     – defaults to `1`

???- note "encode_query"

    Computes sentence embeddings specifically optimized for query representation.

    This method is a specialized version of :meth:`encode` that differs in exactly two ways:  1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" prompt,    if available in the model's ``prompts`` dictionary. 2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.models.Router`    module, it will use the "query" task type to route the input through the appropriate submodules.  .. tip::      If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,     your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks     with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.      Note that :meth:`encode` is the most general method and can be used for any task, including Information     Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three     methods will return identical embeddings.  Args:     sentences (Union[str, List[str]]): The sentences to embed.     prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,         which is either set in the constructor or loaded from the model configuration. For example if         ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What         is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence         is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.     prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the         sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"         because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.     batch_size (int, optional): The batch size used for the computation. Defaults to 32.     show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.     output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:         "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,         to get all output values. Defaults to "sentence_embedding".     precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.         Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.         Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for         reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".     convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.         Defaults to True.     convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.         Defaults to False.     device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:          - A single device string (e.g., "cuda:0", "cpu") for single-process encoding         - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute           encoding across multiple processes         - None to auto-detect available device for single-process encoding         If a list is provided, multi-process encoding will be used. Defaults to None.     normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,         the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.     truncate_dim (int, optional): The dimension to truncate sentence embeddings to.         Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,         i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.         Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference         is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``         from the model initialization is used. Defaults to None.     pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`         for multi-process encoding. If provided, the encoding will be distributed across multiple processes.         This is recommended for large datasets and when multiple GPUs are available. Defaults to None.     chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when         ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.  Returns:     Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.     If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,     a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.  Example:     ::          from sentence_transformers import SentenceTransformer          # Load a pre-trained SentenceTransformer model         model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")          # Encode some queries         queries = [             "What are the effects of climate change?",             "History of artificial intelligence",             "Technical specifications product XYZ",         ]          # Using query-specific encoding         embeddings = model.encode_query(queries)         print(embeddings.shape)         # (3, 768)

    **Parameters**

    - **sentences**     (*'str | list[str] | np.ndarray'*)
    - **prompt_name**     (*'str | None'*)     – defaults to `None`
    - **prompt**     (*'str | None'*)     – defaults to `None`
    - **batch_size**     (*'int'*)     – defaults to `32`
    - **show_progress_bar**     (*'bool | None'*)     – defaults to `None`
    - **output_value**     (*"Literal['sentence_embedding', 'token_embeddings'] | None"*)     – defaults to `sentence_embedding`
    - **precision**     (*"Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']"*)     – defaults to `float32`
    - **convert_to_numpy**     (*'bool'*)     – defaults to `True`
    - **convert_to_tensor**     (*'bool'*)     – defaults to `False`
    - **device**     (*'str | list[str | torch.device] | None'*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.
    - **normalize_embeddings**     (*'bool'*)     – defaults to `False`
    - **truncate_dim**     (*'int | None'*)     – defaults to `None`
        The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is only applicable during inference when :meth:`SentenceTransformer.encode` is called.
    - **pool**     (*"dict[Literal['input', 'output', 'processes'], Any] | None"*)     – defaults to `None`
    - **chunk_size**     (*'int | None'*)     – defaults to `None`
    - **kwargs**

???- note "eval"

    Set the module in evaluation mode.

    This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.  See :ref:`locally-disable-grad-doc` for a comparison between `.eval()` and several similar mechanisms that may be confused with it.  Returns:     Module: self


???- note "evaluate"

    Evaluate the model based on an evaluator

    Args:     evaluator (SentenceEvaluator): The evaluator used to evaluate the model.     output_path (str, optional): The path where the evaluator can write the results. Defaults to None.  Returns:     The evaluation results.

    **Parameters**

    - **evaluator**     (*'SentenceEvaluator'*)
    - **output_path**     (*'str | None'*)     – defaults to `None`

???- note "extend"

    Extends the current Sequential container with layers from another Sequential container.

    Args:     sequential (Sequential): A Sequential container whose layers will be added to the current container.  Example::      >>> import torch.nn as nn     >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))     >>> other = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5))     >>> n.extend(other) # or `n + other`     Sequential(         (0): Linear(in_features=1, out_features=2, bias=True)         (1): Linear(in_features=2, out_features=3, bias=True)         (2): Linear(in_features=3, out_features=4, bias=True)         (3): Linear(in_features=4, out_features=5, bias=True)     )

    **Parameters**

    - **sequential**     (*'Iterable[Module]'*)

???- note "extra_repr"

    Return the extra representation of the module.

    To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.


???- note "fit"

    Deprecated training method from before Sentence Transformers v3.0, it is recommended to use :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method uses :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` behind the scenes, but does not provide as much flexibility as the Trainer itself.

    This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the smallest one to make sure of equal training with each dataset, i.e. round robin sampling.  This method should produce equivalent results in v3.0+ as before v3.0, but if you encounter any issues with your existing training scripts, then you may wish to use :meth:`SentenceTransformer.old_fit <sentence_transformers.SentenceTransformer.old_fit>` instead. That uses the old training method from before v3.0.  Args:     train_objectives: Tuples of (DataLoader, LossFunction). Pass         more than one for multi-task learning     evaluator: An evaluator (sentence_transformers.evaluation)         evaluates the model performance during training on held-         out dev data. It is used to determine the best model         that is saved to disk.     epochs: Number of epochs for training     steps_per_epoch: Number of training steps per epoch. If set         to None (default), one epoch is equal the DataLoader         size from train_objectives.     scheduler: Learning rate scheduler. Available schedulers:         constantlr, warmupconstant, warmuplinear, warmupcosine,         warmupcosinewithhardrestarts     warmup_steps: Behavior depends on the scheduler. For         WarmupLinear (default), the learning rate is increased         from o up to the maximal learning rate. After these many         training steps, the learning rate is decreased linearly         back to zero.     optimizer_class: Optimizer     optimizer_params: Optimizer parameters     weight_decay: Weight decay for model parameters     evaluation_steps: If > 0, evaluate the model using evaluator         after each number of training steps     output_path: Storage path for the model and evaluation files     save_best_model: If true, the best model (according to         evaluator) is stored at output_path     max_grad_norm: Used for gradient normalization.     use_amp: Use Automatic Mixed Precision (AMP). Only for         Pytorch >= 1.6.0     callback: Callback function that is invoked after each         evaluation. It must accept the following three         parameters in this order: `score`, `epoch`, `steps`     show_progress_bar: If True, output a tqdm progress bar     checkpoint_path: Folder to save checkpoints during training     checkpoint_save_steps: Will save a checkpoint after so many         steps     checkpoint_save_total_limit: Total number of checkpoints to         store     resume_from_checkpoint: If true, searches for checkpoints         to continue training from.

    **Parameters**

    - **train_objectives**     (*'Iterable[tuple[DataLoader, nn.Module]]'*)
    - **evaluator**     (*'SentenceEvaluator | None'*)     – defaults to `None`
    - **epochs**     (*'int'*)     – defaults to `1`
    - **steps_per_epoch**     – defaults to `None`
    - **scheduler**     (*'str'*)     – defaults to `WarmupLinear`
    - **warmup_steps**     (*'int'*)     – defaults to `10000`
    - **optimizer_class**     (*'type[Optimizer]'*)     – defaults to `<class 'torch.optim.adamw.AdamW'>`
    - **optimizer_params**     (*'dict[str, object]'*)     – defaults to `{'lr': 2e-05}`
    - **weight_decay**     (*'float'*)     – defaults to `0.01`
    - **evaluation_steps**     (*'int'*)     – defaults to `0`
    - **output_path**     (*'str | None'*)     – defaults to `None`
    - **save_best_model**     (*'bool'*)     – defaults to `True`
    - **max_grad_norm**     (*'float'*)     – defaults to `1`
    - **use_amp**     (*'bool'*)     – defaults to `False`
    - **callback**     (*'Callable[[float, int, int], None]'*)     – defaults to `None`
    - **show_progress_bar**     (*'bool'*)     – defaults to `True`
    - **checkpoint_path**     (*'str | None'*)     – defaults to `None`
    - **checkpoint_save_steps**     (*'int'*)     – defaults to `500`
    - **checkpoint_save_total_limit**     (*'int'*)     – defaults to `0`
    - **resume_from_checkpoint**     (*'bool'*)     – defaults to `False`

???- note "float"

    Casts all floating point parameters and buffers to ``float`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "forward"

    Define the computation performed at every call.

    Should be overridden by all subclasses.  .. note::     Although the recipe for forward pass needs to be defined within     this function, one should call the :class:`Module` instance afterwards     instead of this since the former takes care of running the     registered hooks while the latter silently ignores them.

    **Parameters**

    - **input**     (*'dict[str, Tensor]'*)
    - **kwargs**

???- note "get_adapter_state_dict"

    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

    Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter. If no adapter_name is passed, the active adapter is used.  Args:     *args:         Positional arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict     **kwargs:         Keyword arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict

    **Parameters**

    - **args**
    - **kwargs**

???- note "get_backend"

    Return the backend used for inference, which can be one of "torch", "onnx", or "openvino".

    Returns:     str: The backend used for inference.


???- note "get_buffer"

    Return the buffer given by ``target`` if it exists, otherwise throw an error.

    See the docstring for ``get_submodule`` for a more detailed explanation of this method's functionality as well as how to correctly specify ``target``.  Args:     target: The fully-qualified string name of the buffer         to look for. (See ``get_submodule`` for how to specify a         fully-qualified string.)  Returns:     torch.Tensor: The buffer referenced by ``target``  Raises:     AttributeError: If the target string references an invalid         path or resolves to something that is not a         buffer

    **Parameters**

    - **target**     (*str*)

???- note "get_extra_state"

    Return any extra state to include in the module's state_dict.

    Implement this and a corresponding :func:`set_extra_state` for your module if you need to store extra state. This function is called when building the module's `state_dict()`.  Note that extra state should be picklable to ensure working serialization of the state_dict. We only provide backwards compatibility guarantees for serializing Tensors; other objects may break backwards compatibility if their serialized pickled form changes.  Returns:     object: Any extra state to store in the module's state_dict


???- note "get_max_seq_length"

    Returns the maximal sequence length that the model accepts. Longer inputs will be truncated.

    Returns:     Optional[int]: The maximal sequence length that the model accepts, or None if it is not defined.


???- note "get_model_kwargs"

    Get the keyword arguments specific to this model for the `encode`, `encode_query`, or `encode_document` methods.

    Example:      >>> from sentence_transformers import SentenceTransformer, SparseEncoder     >>> SentenceTransformer("all-MiniLM-L6-v2").get_model_kwargs()     []     >>> SentenceTransformer("jinaai/jina-embeddings-v4", trust_remote_code=True).get_model_kwargs()     ['task', 'truncate_dim']     >>> SparseEncoder("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill").get_model_kwargs()     ['task']  Returns:     list[str]: A list of keyword arguments for the forward pass.


???- note "get_parameter"

    Return the parameter given by ``target`` if it exists, otherwise throw an error.

    See the docstring for ``get_submodule`` for a more detailed explanation of this method's functionality as well as how to correctly specify ``target``.  Args:     target: The fully-qualified string name of the Parameter         to look for. (See ``get_submodule`` for how to specify a         fully-qualified string.)  Returns:     torch.nn.Parameter: The Parameter referenced by ``target``  Raises:     AttributeError: If the target string references an invalid         path or resolves to something that is not an         ``nn.Parameter``

    **Parameters**

    - **target**     (*str*)

???- note "get_sentence_embedding_dimension"

    Returns the number of dimensions in the output of :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.

    Returns:     Optional[int]: The number of dimensions in the output of `encode`. If it's not known, it's `None`.


???- note "get_sentence_features"

???- note "get_submodule"

    Return the submodule given by ``target`` if it exists, otherwise throw an error.

    For example, let's say you have an ``nn.Module`` ``A`` that looks like this:  .. code-block:: text      A(         (net_b): Module(             (net_c): Module(                 (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))             )             (linear): Linear(in_features=100, out_features=200, bias=True)         )     )  (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested submodule ``net_b``, which itself has two submodules ``net_c`` and ``linear``. ``net_c`` then has a submodule ``conv``.)  To check whether or not we have the ``linear`` submodule, we would call ``get_submodule("net_b.linear")``. To check whether we have the ``conv`` submodule, we would call ``get_submodule("net_b.net_c.conv")``.  The runtime of ``get_submodule`` is bounded by the degree of module nesting in ``target``. A query against ``named_modules`` achieves the same result, but it is O(N) in the number of transitive modules. So, for a simple check to see if some submodule exists, ``get_submodule`` should always be used.  Args:     target: The fully-qualified string name of the submodule         to look for. (See above example for how to specify a         fully-qualified string.)  Returns:     torch.nn.Module: The submodule referenced by ``target``  Raises:     AttributeError: If at any point along the path resulting from         the target string the (sub)path resolves to a non-existent         attribute name or an object that is not an instance of ``nn.Module``.

    **Parameters**

    - **target**     (*str*)

???- note "gradient_checkpointing_enable"

???- note "half"

    Casts all floating point parameters and buffers to ``half`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "has_peft_compatible_model"

???- note "insert"

    Inserts a module into the Sequential container at the specified index.

    Args:     index (int): The index to insert the module.     module (Module): The module to be inserted.  Example::      >>> import torch.nn as nn     >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))     >>> n.insert(0, nn.Linear(3, 4))     Sequential(         (0): Linear(in_features=3, out_features=4, bias=True)         (1): Linear(in_features=1, out_features=2, bias=True)         (2): Linear(in_features=2, out_features=3, bias=True)     )

    **Parameters**

    - **index**     (*'int'*)
    - **module**     (*'Module'*)

???- note "insert_prefix_token"

    Inserts a prefix token at the beginning of each sequence in the input tensor.

    **Parameters**

    - **input_ids**     (*'torch.Tensor'*)
    - **prefix_id**     (*'int'*)

???- note "ipu"

    Move all model parameters and buffers to the IPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on IPU while being optimized.  .. note::     This method modifies the module in-place.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[torch.device, int, NoneType]*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.

???- note "load"

???- note "load_adapter"

    Load adapter weights from file or remote Hub folder." If you are not familiar with adapters and PEFT methods, we invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

    Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.  Args:     *args:         Positional arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter     **kwargs:         Keyword arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter

    **Parameters**

    - **args**
    - **kwargs**

???- note "load_state_dict"

    Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

    If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` function.  .. warning::     If :attr:`assign` is ``True`` the optimizer must be created after     the call to :attr:`load_state_dict` unless     :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.  Args:     state_dict (dict): a dict containing parameters and         persistent buffers.     strict (bool, optional): whether to strictly enforce that the keys         in :attr:`state_dict` match the keys returned by this module's         :meth:`~torch.nn.Module.state_dict` function. Default: ``True``     assign (bool, optional): When set to ``False``, the properties of the tensors         in the current module are preserved whereas setting it to ``True`` preserves         properties of the Tensors in the state dict. The only         exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`         for which the value from the module is preserved. Default: ``False``  Returns:     ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:         * ``missing_keys`` is a list of str containing any keys that are expected             by this module but missing from the provided ``state_dict``.         * ``unexpected_keys`` is a list of str containing the keys that are not             expected by this module but present in the provided ``state_dict``.  Note:     If a parameter or buffer is registered as ``None`` and its corresponding key     exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a     ``RuntimeError``.

    **Parameters**

    - **state_dict**     (*collections.abc.Mapping[str, typing.Any]*)
    - **strict**     (*bool*)     – defaults to `True`
    - **assign**     (*bool*)     – defaults to `False`

???- note "model_card_data_class"

    A dataclass storing data used in the model card.

    Args:     language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,         e.g. "en" or ["en", "de", "nl"]     license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",         or "cc-by-nc-sa-4.0"     model_name (`Optional[str]`): The pretty name of the model, e.g. "SentenceTransformer based on microsoft/mpnet-base".     model_id (`Optional[str]`): The model ID when pushing the model to the Hub,         e.g. "tomaarsen/sbert-mpnet-base-allnli".     train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.         e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]     eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.         e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]     task_name (`str`): The human-readable task the model is trained on,         e.g. "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".     tags (`Optional[List[str]]`): A list of tags for the model,         e.g. ["sentence-transformers", "sentence-similarity", "feature-extraction"].     local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.         Defaults to False.     generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset,         and compute their similarities. Defaults to True.  .. tip::      Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and     include it in your model cards.  Example::      >>> model = SentenceTransformer(     ...     "microsoft/mpnet-base",     ...     model_card_data=SentenceTransformerModelCardData(     ...         model_id="tomaarsen/sbert-mpnet-base-allnli",     ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],     ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],     ...         license="apache-2.0",     ...         language="en",     ...     ),     ... )

    **Parameters**

    - **language**     (*'str | list[str] | None'*)     – defaults to `<factory>`
    - **license**     (*'str | None'*)     – defaults to `None`
    - **model_name**     (*'str | None'*)     – defaults to `None`
    - **model_id**     (*'str | None'*)     – defaults to `None`
    - **train_datasets**     (*'list[dict[str, str]]'*)     – defaults to `<factory>`
    - **eval_datasets**     (*'list[dict[str, str]]'*)     – defaults to `<factory>`
    - **task_name**     (*'str'*)     – defaults to `semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more`
    - **tags**     (*'list[str] | None'*)     – defaults to `<factory>`
    - **local_files_only**     (*'bool'*)     – defaults to `False`
        Whether or not to only look at local files (i.e., do not try to download the model).
    - **generate_widget_examples**     (*'bool'*)     – defaults to `True`

???- note "modules"

    Return an iterator over all modules in the network.

    Yields:     Module: a module in the network  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.modules()):     ...     print(idx, '->', m)      0 -> Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )     1 -> Linear(in_features=2, out_features=2, bias=True)


???- note "mtia"

    Move all model parameters and buffers to the MTIA.

    This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on MTIA while being optimized.  .. note::     This method modifies the module in-place.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[torch.device, int, NoneType]*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.

???- note "named_buffers"

    Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

    Args:     prefix (str): prefix to prepend to all buffer names.     recurse (bool, optional): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module. Defaults to True.     remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.  Yields:     (str, torch.Tensor): Tuple containing the name and buffer  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, buf in self.named_buffers():     >>>     if name in ['running_var']:     >>>         print(buf.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``
    - **recurse**     (*bool*)     – defaults to `True`
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "named_children"

    Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

    Yields:     (str, Module): Tuple containing a name and child module  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, module in model.named_children():     >>>     if name in ['conv4', 'conv5']:     >>>         print(module)


???- note "named_modules"

    Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

    Args:     memo: a memo to store the set of modules already added to the result     prefix: a prefix that will be added to the name of the module     remove_duplicate: whether to remove the duplicated module instances in the result         or not  Yields:     (str, Module): Tuple of name and module  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.named_modules()):     ...     print(idx, '->', m)      0 -> ('', Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     ))     1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

    **Parameters**

    - **memo**     (*Optional[set['Module']]*)     – defaults to `None`
    - **prefix**     (*str*)     – defaults to ``
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "named_parameters"

    Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

    Args:     prefix (str): prefix to prepend to all parameter names.     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.     remove_duplicate (bool, optional): whether to remove the duplicated         parameters in the result. Defaults to True.  Yields:     (str, Parameter): Tuple containing the name and parameter  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, param in self.named_parameters():     >>>     if name in ['bias']:     >>>         print(param.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``
    - **recurse**     (*bool*)     – defaults to `True`
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "old_fit"

    Deprecated training method from before Sentence Transformers v3.0, it is recommended to use :class:`sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method should only be used if you encounter issues with your existing training scripts after upgrading to v3.0+.

    This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the smallest one to make sure of equal training with each dataset, i.e. round robin sampling.  Args:     train_objectives: Tuples of (DataLoader, LossFunction). Pass         more than one for multi-task learning     evaluator: An evaluator (sentence_transformers.evaluation)         evaluates the model performance during training on held-         out dev data. It is used to determine the best model         that is saved to disk.     epochs: Number of epochs for training     steps_per_epoch: Number of training steps per epoch. If set         to None (default), one epoch is equal the DataLoader         size from train_objectives.     scheduler: Learning rate scheduler. Available schedulers:         constantlr, warmupconstant, warmuplinear, warmupcosine,         warmupcosinewithhardrestarts     warmup_steps: Behavior depends on the scheduler. For         WarmupLinear (default), the learning rate is increased         from o up to the maximal learning rate. After these many         training steps, the learning rate is decreased linearly         back to zero.     optimizer_class: Optimizer     optimizer_params: Optimizer parameters     weight_decay: Weight decay for model parameters     evaluation_steps: If > 0, evaluate the model using evaluator         after each number of training steps     output_path: Storage path for the model and evaluation files     save_best_model: If true, the best model (according to         evaluator) is stored at output_path     max_grad_norm: Used for gradient normalization.     use_amp: Use Automatic Mixed Precision (AMP). Only for         Pytorch >= 1.6.0     callback: Callback function that is invoked after each         evaluation. It must accept the following three         parameters in this order: `score`, `epoch`, `steps`     show_progress_bar: If True, output a tqdm progress bar     checkpoint_path: Folder to save checkpoints during training     checkpoint_save_steps: Will save a checkpoint after so many         steps     checkpoint_save_total_limit: Total number of checkpoints to         store

    **Parameters**

    - **train_objectives**     (*'Iterable[tuple[DataLoader, nn.Module]]'*)
    - **evaluator**     (*'SentenceEvaluator | None'*)     – defaults to `None`
    - **epochs**     (*'int'*)     – defaults to `1`
    - **steps_per_epoch**     – defaults to `None`
    - **scheduler**     (*'str'*)     – defaults to `WarmupLinear`
    - **warmup_steps**     (*'int'*)     – defaults to `10000`
    - **optimizer_class**     (*'type[Optimizer]'*)     – defaults to `<class 'torch.optim.adamw.AdamW'>`
    - **optimizer_params**     (*'dict[str, object]'*)     – defaults to `{'lr': 2e-05}`
    - **weight_decay**     (*'float'*)     – defaults to `0.01`
    - **evaluation_steps**     (*'int'*)     – defaults to `0`
    - **output_path**     (*'str | None'*)     – defaults to `None`
    - **save_best_model**     (*'bool'*)     – defaults to `True`
    - **max_grad_norm**     (*'float'*)     – defaults to `1`
    - **use_amp**     (*'bool'*)     – defaults to `False`
    - **callback**     (*'Callable[[float, int, int], None]'*)     – defaults to `None`
    - **show_progress_bar**     (*'bool'*)     – defaults to `True`
    - **checkpoint_path**     (*'str | None'*)     – defaults to `None`
    - **checkpoint_save_steps**     (*'int'*)     – defaults to `500`
    - **checkpoint_save_total_limit**     (*'int'*)     – defaults to `0`

???- note "parameters"

    Return an iterator over module parameters.

    This is typically passed to an optimizer.  Args:     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.  Yields:     Parameter: module parameter  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for param in model.parameters():     >>>     print(type(param), param.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`

???- note "pool_embeddings_hierarchical"

    Pools the embeddings hierarchically by clustering and averaging them.

    **Parameters**

    - **documents_embeddings**     (*'list[torch.Tensor]'*)
    - **pool_factor**     (*'int'*)     – defaults to `1`
    - **protected_tokens**     (*'int'*)     – defaults to `1`

    **Returns**

    *list[torch.Tensor]*:     A list of pooled embeddings for each document.

???- note "pop"

???- note "push_to_hub"

    Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

    Args:     repo_id (str): Repository name for your model in the Hub, including the user or organization.     token (str, optional): An authentication token (See https://huggingface.co/settings/token)     private (bool, optional): Set to true, for hosting a private model     safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way     commit_message (str, optional): Message to commit while pushing.     local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded     exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible     replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card     train_datasets (List[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.     revision (str, optional): Branch to push the uploaded files to     create_pr (bool, optional): If True, create a pull request instead of pushing directly to the main branch  Returns:     str: The url of the commit of your model in the repository on the Hugging Face Hub.

    **Parameters**

    - **repo_id**     (*'str'*)
    - **token**     (*'str | None'*)     – defaults to `None`
        Hugging Face authentication token to download private models.
    - **private**     (*'bool | None'*)     – defaults to `None`
    - **safe_serialization**     (*'bool'*)     – defaults to `True`
    - **commit_message**     (*'str | None'*)     – defaults to `None`
    - **local_model_path**     (*'str | None'*)     – defaults to `None`
    - **exist_ok**     (*'bool'*)     – defaults to `False`
    - **replace_model_card**     (*'bool'*)     – defaults to `False`
    - **train_datasets**     (*'list[str] | None'*)     – defaults to `None`
    - **revision**     (*'str | None'*)     – defaults to `None`
        The specific model version to use. It can be a branch name, a tag name, or a commit id, for a stored model on Hugging Face.
    - **create_pr**     (*'bool'*)     – defaults to `False`

???- note "register_backward_hook"

    Register a backward hook on the module.

    This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and the behavior of this function will change in future versions.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]*)

???- note "register_buffer"

    Add a buffer to the module.

    This is typically used to register a buffer that should not be considered a model parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state. Buffers, by default, are persistent and will be saved alongside parameters. This behavior can be changed by setting :attr:`persistent` to ``False``. The only difference between a persistent buffer and a non-persistent buffer is that the latter will not be a part of this module's :attr:`state_dict`.  Buffers can be accessed as attributes using given names.  Args:     name (str): name of the buffer. The buffer can be accessed         from this module using the given name     tensor (Tensor or None): buffer to be registered. If ``None``, then operations         that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,         the buffer is **not** included in the module's :attr:`state_dict`.     persistent (bool): whether the buffer is part of this module's         :attr:`state_dict`.  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> self.register_buffer('running_mean', torch.zeros(num_features))

    **Parameters**

    - **name**     (*str*)
    - **tensor**     (*Optional[torch.Tensor]*)
    - **persistent**     (*bool*)     – defaults to `True`

???- note "register_forward_hook"

    Register a forward hook on the module.

    The hook will be called every time after :func:`forward` has computed an output.  If ``with_kwargs`` is ``False`` or not specified, the input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the output. It can modify the input inplace but it will not have effect on forward since this is called after :func:`forward` is called. The hook should have the following signature::      hook(module, args, output) -> None or modified output  If ``with_kwargs`` is ``True``, the forward hook will be passed the ``kwargs`` given to the forward function and be expected to return the output possibly modified. The hook should have the following signature::      hook(module, args, kwargs, output) -> None or modified output  Args:     hook (Callable): The user defined hook to be registered.     prepend (bool): If ``True``, the provided ``hook`` will be fired         before all existing ``forward`` hooks on this         :class:`torch.nn.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``forward`` hooks on         this :class:`torch.nn.Module`. Note that global         ``forward`` hooks registered with         :func:`register_module_forward_hook` will fire before all hooks         registered by this method.         Default: ``False``     with_kwargs (bool): If ``True``, the ``hook`` will be passed the         kwargs given to the forward function.         Default: ``False``     always_call (bool): If ``True`` the ``hook`` will be run regardless of         whether an exception is raised while calling the Module.         Default: ``False``  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]]*)
    - **prepend**     (*bool*)     – defaults to `False`
    - **with_kwargs**     (*bool*)     – defaults to `False`
    - **always_call**     (*bool*)     – defaults to `False`

???- note "register_forward_pre_hook"

    Register a forward pre-hook on the module.

    The hook will be called every time before :func:`forward` is invoked.  If ``with_kwargs`` is false or not specified, the input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the input. User can either return a tuple or a single modified value in the hook. We will wrap the value into a tuple if a single value is returned (unless that value is already a tuple). The hook should have the following signature::      hook(module, args) -> None or modified input  If ``with_kwargs`` is true, the forward pre-hook will be passed the kwargs given to the forward function. And if the hook modifies the input, both the args and kwargs should be returned. The hook should have the following signature::      hook(module, args, kwargs) -> None or a tuple of modified input and kwargs  Args:     hook (Callable): The user defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``forward_pre`` hooks on this         :class:`torch.nn.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``forward_pre`` hooks         on this :class:`torch.nn.Module`. Note that global         ``forward_pre`` hooks registered with         :func:`register_module_forward_pre_hook` will fire before all         hooks registered by this method.         Default: ``False``     with_kwargs (bool): If true, the ``hook`` will be passed the kwargs         given to the forward function.         Default: ``False``  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]]*)
    - **prepend**     (*bool*)     – defaults to `False`
    - **with_kwargs**     (*bool*)     – defaults to `False`

???- note "register_full_backward_hook"

    Register a backward hook on the module.

    The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:      1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.     2. If none of the module inputs require gradients, the hook will fire when the gradients are computed        with respect to module outputs.     3. If none of the module outputs require gradients, then the hooks will not fire.  The hook should have the following signature::      hook(module, grad_input, grad_output) -> tuple(Tensor) or None  The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients with respect to the inputs and outputs respectively. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the input that will be used in place of :attr:`grad_input` in subsequent computations. :attr:`grad_input` will only correspond to the inputs given as positional arguments and all kwarg arguments are ignored. Entries in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor arguments.  For technical reasons, when this hook is applied to a Module, its forward function will receive a view of each Tensor passed to the Module. Similarly the caller will receive a view of each Tensor returned by the Module's forward function.  .. warning ::     Modifying inputs or outputs inplace is not allowed when using backward hooks and     will raise an error.  Args:     hook (Callable): The user-defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``backward`` hooks on this         :class:`torch.nn.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``backward`` hooks on         this :class:`torch.nn.Module`. Note that global         ``backward`` hooks registered with         :func:`register_module_full_backward_hook` will fire before         all hooks registered by this method.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]*)
    - **prepend**     (*bool*)     – defaults to `False`

???- note "register_full_backward_pre_hook"

    Register a backward pre-hook on the module.

    The hook will be called every time the gradients for the module are computed. The hook should have the following signature::      hook(module, grad_output) -> tuple[Tensor] or None  The :attr:`grad_output` is a tuple. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the output that will be used in place of :attr:`grad_output` in subsequent computations. Entries in :attr:`grad_output` will be ``None`` for all non-Tensor arguments.  For technical reasons, when this hook is applied to a Module, its forward function will receive a view of each Tensor passed to the Module. Similarly the caller will receive a view of each Tensor returned by the Module's forward function.  .. warning ::     Modifying inputs inplace is not allowed when using backward hooks and     will raise an error.  Args:     hook (Callable): The user-defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``backward_pre`` hooks on this         :class:`torch.nn.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``backward_pre`` hooks         on this :class:`torch.nn.Module`. Note that global         ``backward_pre`` hooks registered with         :func:`register_module_full_backward_pre_hook` will fire before         all hooks registered by this method.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]*)
    - **prepend**     (*bool*)     – defaults to `False`

???- note "register_load_state_dict_post_hook"

    Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

    It should have the following signature::     hook(module, incompatible_keys) -> None  The ``module`` argument is the current module that this hook is registered on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys`` is a ``list`` of ``str`` containing the missing keys and ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.  The given incompatible_keys can be modified inplace if needed.  Note that the checks performed when calling :func:`load_state_dict` with ``strict=True`` are affected by modifications the hook makes to ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either set of keys will result in an error being thrown when ``strict=True``, and clearing out both missing and unexpected keys will avoid an error.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**

???- note "register_load_state_dict_pre_hook"

    Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

    It should have the following signature::     hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950  Arguments:     hook (Callable): Callable hook that will be invoked before         loading the state dict.

    **Parameters**

    - **hook**

???- note "register_module"

    Alias for :func:`add_module`.

    **Parameters**

    - **name**     (*str*)
    - **module**     (*Optional[ForwardRef('Module')]*)

???- note "register_parameter"

    Add a parameter to the module.

    The parameter can be accessed as an attribute using given name.  Args:     name (str): name of the parameter. The parameter can be accessed         from this module using the given name     param (Parameter or None): parameter to be added to the module. If         ``None``, then operations that run on parameters, such as :attr:`cuda`,         are ignored. If ``None``, the parameter is **not** included in the         module's :attr:`state_dict`.

    **Parameters**

    - **name**     (*str*)
    - **param**     (*Optional[torch.nn.parameter.Parameter]*)

???- note "register_state_dict_post_hook"

    Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

    It should have the following signature::     hook(module, state_dict, prefix, local_metadata) -> None  The registered hooks can modify the ``state_dict`` inplace.

    **Parameters**

    - **hook**

???- note "register_state_dict_pre_hook"

    Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

    It should have the following signature::     hook(module, prefix, keep_vars) -> None  The registered hooks can be used to perform pre-processing before the ``state_dict`` call is made.

    **Parameters**

    - **hook**

???- note "requires_grad_"

    Change if autograd should record operations on parameters in this module.

    This method sets the parameters' :attr:`requires_grad` attributes in-place.  This method is helpful for freezing part of the module for finetuning or training parts of a model individually (e.g., GAN training).  See :ref:`locally-disable-grad-doc` for a comparison between `.requires_grad_()` and several similar mechanisms that may be confused with it.  Args:     requires_grad (bool): whether autograd should record operations on                           parameters in this module. Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **requires_grad**     (*bool*)     – defaults to `True`

???- note "save"

    Saves a model and its configuration files to a directory, so that it can be loaded with ``SentenceTransformer(path)`` again.

    Args:     path (str): Path on disc where the model will be saved.     model_name (str, optional): Optional model name.     create_model_card (bool, optional): If True, create a README.md with basic information about this model.     train_datasets (list[str], optional): Optional list with the names of the datasets used to train the model.     safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model         the traditional (but unsafe) PyTorch way.

    **Parameters**

    - **path**     (*'str'*)
    - **model_name**     (*'str | None'*)     – defaults to `None`
    - **create_model_card**     (*'bool'*)     – defaults to `True`
    - **train_datasets**     (*'list[str] | None'*)     – defaults to `None`
    - **safe_serialization**     (*'bool'*)     – defaults to `True`

???- note "save_pretrained"

    Saves a model and its configuration files to a directory, so that it can be loaded with ``SentenceTransformer(path)`` again.

    Args:     path (str): Path on disk where the model will be saved.     model_name (str, optional): Optional model name.     create_model_card (bool, optional): If True, create a README.md with basic information about this model.     train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.     safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model         the traditional (but unsafe) PyTorch way.

    **Parameters**

    - **path**     (*'str'*)
    - **model_name**     (*'str | None'*)     – defaults to `None`
    - **create_model_card**     (*'bool'*)     – defaults to `True`
    - **train_datasets**     (*'list[str] | None'*)     – defaults to `None`
    - **safe_serialization**     (*'bool'*)     – defaults to `True`

???- note "save_to_hub"

    DEPRECATED, use `push_to_hub` instead.

    Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.  Args:     repo_id (str): Repository name for your model in the Hub, including the user or organization.     token (str, optional): An authentication token (See https://huggingface.co/settings/token)     private (bool, optional): Set to true, for hosting a private model     safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way     commit_message (str, optional): Message to commit while pushing.     local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded     exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible     replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card     train_datasets (List[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.  Returns:     str: The url of the commit of your model in the repository on the Hugging Face Hub.

    **Parameters**

    - **repo_id**     (*'str'*)
    - **organization**     (*'str | None'*)     – defaults to `None`
    - **token**     (*'str | None'*)     – defaults to `None`
        Hugging Face authentication token to download private models.
    - **private**     (*'bool | None'*)     – defaults to `None`
    - **safe_serialization**     (*'bool'*)     – defaults to `True`
    - **commit_message**     (*'str'*)     – defaults to `Add new SentenceTransformer model.`
    - **local_model_path**     (*'str | None'*)     – defaults to `None`
    - **exist_ok**     (*'bool'*)     – defaults to `False`
    - **replace_model_card**     (*'bool'*)     – defaults to `False`
    - **train_datasets**     (*'list[str] | None'*)     – defaults to `None`

???- note "set_adapter"

    Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

    Args:     *args:         Positional arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter     **kwargs:         Keyword arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation         https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter

    **Parameters**

    - **args**
    - **kwargs**

???- note "set_extra_state"

    Set extra state contained in the loaded `state_dict`.

    This function is called from :func:`load_state_dict` to handle any extra state found within the `state_dict`. Implement this function and a corresponding :func:`get_extra_state` for your module if you need to store extra state within its `state_dict`.  Args:     state (dict): Extra state from the `state_dict`

    **Parameters**

    - **state**     (*Any*)

???- note "set_pooling_include_prompt"

    Sets the `include_prompt` attribute in the pooling layer in the model, if there is one.

    This is useful for INSTRUCTOR models, as the prompt should be excluded from the pooling strategy for these models.  Args:     include_prompt (bool): Whether to include the prompt in the pooling layer.  Returns:     None

    **Parameters**

    - **include_prompt**     (*'bool'*)

???- note "set_submodule"

    Set the submodule given by ``target`` if it exists, otherwise throw an error.

    .. note::     If ``strict`` is set to ``False`` (default), the method will replace an existing submodule     or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,     the method will only attempt to replace an existing submodule and throw an error if     the submodule does not exist.  For example, let's say you have an ``nn.Module`` ``A`` that looks like this:  .. code-block:: text      A(         (net_b): Module(             (net_c): Module(                 (conv): Conv2d(3, 3, 3)             )             (linear): Linear(3, 3)         )     )  (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested submodule ``net_b``, which itself has two submodules ``net_c`` and ``linear``. ``net_c`` then has a submodule ``conv``.)  To override the ``Conv2d`` with a new submodule ``Linear``, you could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))`` where ``strict`` could be ``True`` or ``False``  To add a new submodule ``Conv2d`` to the existing ``net_b`` module, you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.  In the above if you set ``strict=True`` and call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError will be raised because ``net_b`` does not have a submodule named ``conv``.  Args:     target: The fully-qualified string name of the submodule         to look for. (See above example for how to specify a         fully-qualified string.)     module: The module to set the submodule to.     strict: If ``False``, the method will replace an existing submodule         or create a new submodule if the parent module exists. If ``True``,         the method will only attempt to replace an existing submodule and throw an error         if the submodule doesn't already exist.  Raises:     ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.     AttributeError: If at any point along the path resulting from         the ``target`` string the (sub)path resolves to a non-existent         attribute name or an object that is not an instance of ``nn.Module``.

    **Parameters**

    - **target**     (*str*)
    - **module**     (*'Module'*)
    - **strict**     (*bool*)     – defaults to `False`

???- note "share_memory"

    See :meth:`torch.Tensor.share_memory_`.


???- note "skiplist_mask"

    Create a mask for the set of input_ids that are in the skiplist.

    **Parameters**

    - **input_ids**     (*'torch.Tensor'*)
    - **skiplist**     (*'list[int]'*)

???- note "smart_batching_collate"

    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model Here, batch is a list of InputExample instances: [InputExample(...), ...]

    Args:     batch: a batch from a SmartBatchingDataset  Returns:     a batch of tensors for the model

    **Parameters**

    - **batch**     (*'list[InputExample]'*)

???- note "start_multi_process_pool"

    Starts a multi-process pool to process the encoding with several independent processes. This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised to start only one process per GPU. This method works together with encode_multi_process and stop_multi_process_pool.

    **Parameters**

    - **target_devices**     (*'list[str]'*)     – defaults to `None`

    **Returns**

    *dict*:     A dictionary with the target processes, an input queue, and an output queue.

???- note "state_dict"

    Return a dictionary containing references to the whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are included. Keys are corresponding parameter and buffer names. Parameters and buffers set to ``None`` are not included.  .. note::     The returned object is a shallow copy. It contains references     to the module's parameters and buffers.  .. warning::     Currently ``state_dict()`` also accepts positional arguments for     ``destination``, ``prefix`` and ``keep_vars`` in order. However,     this is being deprecated and keyword arguments will be enforced in     future releases.  .. warning::     Please avoid the use of argument ``destination`` as it is not     designed for end-users.  Args:     destination (dict, optional): If provided, the state of module will         be updated into the dict and the same object is returned.         Otherwise, an ``OrderedDict`` will be created and returned.         Default: ``None``.     prefix (str, optional): a prefix added to parameter and buffer         names to compose the keys in state_dict. Default: ``''``.     keep_vars (bool, optional): by default the :class:`~torch.Tensor` s         returned in the state dict are detached from autograd. If it's         set to ``True``, detaching will not be performed.         Default: ``False``.  Returns:     dict:         a dictionary containing a whole state of the module  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> module.state_dict().keys()     ['bias', 'weight']

    **Parameters**

    - **args**
    - **destination**     – defaults to `None`
    - **prefix**     – defaults to ``
    - **keep_vars**     – defaults to `False`

???- note "stop_multi_process_pool"

    Stops all processes started with start_multi_process_pool.

    Args:     pool (Dict[str, object]): A dictionary containing the input queue, output queue, and process list.  Returns:     None

    - **pool**     (*"dict[Literal['input', 'output', 'processes'], Any]"*)

???- note "to"

    Move and/or cast the parameters and buffers.

    This can be called as  .. function:: to(device=None, dtype=None, non_blocking=False)    :noindex:  .. function:: to(dtype, non_blocking=False)    :noindex:  .. function:: to(tensor, non_blocking=False)    :noindex:  .. function:: to(memory_format=torch.channels_last)    :noindex:  Its signature is similar to :meth:`torch.Tensor.to`, but only accepts floating point or complex :attr:`dtype`\ s. In addition, this method will only cast the floating point or complex parameters and buffers to :attr:`dtype` (if given). The integral parameters and buffers will be moved :attr:`device`, if that is given, but with dtypes unchanged. When :attr:`non_blocking` is set, it tries to convert/move asynchronously with respect to the host if possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.  See below for examples.  .. note::     This method modifies the module in-place.  Args:     device (:class:`torch.device`): the desired device of the parameters         and buffers in this module     dtype (:class:`torch.dtype`): the desired floating point or complex dtype of         the parameters and buffers in this module     tensor (torch.Tensor): Tensor whose dtype and device are the desired         dtype and device for all parameters and buffers in this module     memory_format (:class:`torch.memory_format`): the desired memory         format for 4D parameters and buffers in this module (keyword         only argument)  Returns:     Module: self  Examples::      >>> # xdoctest: +IGNORE_WANT("non-deterministic")     >>> linear = nn.Linear(2, 2)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]])     >>> linear.to(torch.double)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]], dtype=torch.float64)     >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)     >>> gpu1 = torch.device("cuda:1")     >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')     >>> cpu = torch.device("cpu")     >>> linear.to(cpu)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16)      >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)     >>> linear.weight     Parameter containing:     tensor([[ 0.3741+0.j,  0.2382+0.j],             [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)     >>> linear(torch.ones(3, 2, dtype=torch.cdouble))     tensor([[0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

    **Parameters**

    - **args**
    - **kwargs**

???- note "to_empty"

    Move the parameters and buffers to the specified device without copying storage.

    Args:     device (:class:`torch.device`): The desired device of the parameters         and buffers in this module.     recurse (bool): Whether parameters and buffers of submodules should         be recursively moved to the specified device.  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, str, torch.device, NoneType]*)
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.
    - **recurse**     (*bool*)     – defaults to `True`

???- note "tokenize"

    Tokenizes the input texts.

    Args:     texts (Union[list[str], list[dict], list[tuple[str, str]]]): A list of texts to be tokenized.     is_query (bool): Flag to indicate if the texts are queries. Defaults to True.     pad (bool): Flag to indicate if elements should be padded to max length. Defaults to False.  Returns:     dict[str, torch.Tensor]: A dictionary of tensors with the tokenized texts, including "input_ids",         "attention_mask", and optionally "token_type_ids".

    **Parameters**

    - **texts**     (*'list[str] | list[dict] | list[tuple[str, str]]'*)
    - **is_query**     (*'bool'*)     – defaults to `True`
    - **pad**     (*'bool'*)     – defaults to `False`

???- note "train"

    Set the module in training mode.

    This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  Args:     mode (bool): whether to set training mode (``True``) or evaluation                  mode (``False``). Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **mode**     (*bool*)     – defaults to `True`

???- note "truncate_sentence_embeddings"

    In this context, :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` outputs sentence embeddings truncated at dimension ``truncate_dim``.

    This may be useful when you are using the same model for different applications where different dimensions are needed.  Args:     truncate_dim (int, optional): The dimension to truncate sentence embeddings to. ``None`` does no truncation.  Example:     ::          from sentence_transformers import SentenceTransformer          model = SentenceTransformer("all-mpnet-base-v2")          with model.truncate_sentence_embeddings(truncate_dim=16):             embeddings_truncated = model.encode(["hello there", "hiya"])         assert embeddings_truncated.shape[-1] == 16

    **Parameters**

    - **truncate_dim**     (*'int | None'*)
        The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is only applicable during inference when :meth:`SentenceTransformer.encode` is called.

???- note "type"

    Casts all parameters and buffers to :attr:`dst_type`.

    .. note::     This method modifies the module in-place.  Args:     dst_type (type or string): the desired type  Returns:     Module: self

    **Parameters**

    - **dst_type**     (*Union[torch.dtype, str]*)

???- note "xpu"

    Move all model parameters and buffers to the XPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on XPU while being optimized.  .. note::     This method modifies the module in-place.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[torch.device, int, NoneType]*)     – defaults to `None`
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.

???- note "zero_grad"

    Reset gradients of all model parameters.

    See similar function under :class:`torch.optim.Optimizer` for more context.  Args:     set_to_none (bool): instead of setting to zero, set the grads to None.         See :meth:`torch.optim.Optimizer.zero_grad` for details.

    **Parameters**

    - **set_to_none**     (*bool*)     – defaults to `True`
