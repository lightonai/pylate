from __future__ import annotations

import copy
import json
import logging
import math
import os
import string
from typing import Iterable, Literal, Optional

import numpy as np
import torch
from numpy import ndarray
from scipy.cluster import hierarchy
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense as DenseSentenceTransformer
from sentence_transformers.models import Transformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import batch_to_device, load_file_path
from torch import nn
from tqdm.autonotebook import trange
from transformers.utils import cached_file

from ..hf_hub.model_card import PylateModelCardData
from ..scores import SimilarityFunction
from ..utils import _start_multi_process_pool
from .Dense import Dense

logger = logging.getLogger(__name__)


__version__ = "3.0.0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"


class ColBERT(SentenceTransformer):
    """
    Loads or creates a ColBERT model that can be used to map sentences / text to multi-vectors embeddings.

    Parameters
    ----------
    model_name_or_path
        If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download
        a pre-trained SentenceTransformer model. If that fails, tries to construct a model from the Hugging Face Hub
        with that name.
    modules
        A list of torch Modules that should be called sequentially, can be used to create custom SentenceTransformer
        models from scratch.
    device
        Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
        can be used.
    prompts
        A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text. The prompt
        text will be prepended before any text to encode. For example:
        `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
        titles in "}`.
    default_prompt_name
        The name of the prompt that should be used by default. If not set, no prompt will be applied.
    similarity_fn_name
        The name of the similarity function to use. Valid options are "cosine", "dot", "euclidean", and "manhattan".
        If not set, it is automatically set to "cosine" if `similarity` or `similarity_pairwise` are called while
        `model.similarity_fn_name` is still `None`.
    cache_folder
        Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
    trust_remote_code
        Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should
        only be set to True for repositories you trust and in which you have read the code, as it will execute code
        present on the Hub on your local machine.
    revision
        The specific model version to use. It can be a branch name, a tag name, or a commit id, for a stored model on
        Hugging Face.
    local_files_only
        Whether or not to only look at local files (i.e., do not try to download the model).
    token
        Hugging Face authentication token to download private models.
    use_auth_token
        Deprecated argument. Please use `token` instead.
    truncate_dim
        The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is only applicable
        during inference when :meth:`SentenceTransformer.encode` is called.
    embedding_size
        The output size of the projection layer. Default to 128.
    query_prefix
        Prefix to add to the queries.
    document_prefix
        Prefix to add to the documents.
    add_special_tokens
        Add the prefix to the inputs.
    truncation
        Truncate the inputs to the encoder max lengths or use sliding window encoding.
    query_length
        The length of the query to truncate/pad to with mask tokens. If set, will override the config value. Default to 32.
    document_length
        The max length of the document to truncate. If set, will override the config value. Default to 180.
    attend_to_expansion_tokens
        Whether to attend to the expansion tokens in the attention layers model. If False, the original tokens will
        not only attend to the expansion tokens, only the expansion tokens will attend to the original tokens. Default
        is False (as in the original ColBERT codebase).
    skiplist_words
        A list of words to skip from the documents scoring (note that these tokens are used for encoding and are only skipped during the scoring). Default is the list of string.punctuation.
    model_kwargs : dict, optional
        Additional model configuration parameters to be passed to the Huggingface Transformers model. Particularly
        useful options are:

        - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`. The
            different options are:

                1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified ``dtype``,
                ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will get
                loaded in ``torch.float`` (fp32).

                2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be attempted
                to be used. If this entry isn't found then next check the ``dtype`` of the first weight in the
                checkpoint that's of a floating point type and use that as ``dtype``. This will load the model using
                the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how the
                model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
        - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
            `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
            <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
            or `"flash_attention_2"` (using `Dao-AILab/flash-attention
            <https://github.com/Dao-AILab/flash-attention>`_). By default, if available, SDPA will be used for
            torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

        See the `PreTrainedModel.from_pretrained
        <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
        documentation for more details.
    tokenizer_kwargs
        Additional tokenizer configuration parameters to be passed to the Huggingface Transformers tokenizer. See the
        `AutoTokenizer.from_pretrained
        <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
        documentation for more details.
    config_kwargs
        Additional model configuration parameters to be passed to the Huggingface Transformers config. See the
        `AutoConfig.from_pretrained
        <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
        documentation for more details.
    model_card_data
        A model card data object that contains information about the model. This is used to generate a model card when
        saving the model. If not set, a default model card data object is created.

    Examples
    --------
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

    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: Optional[str | SimilarityFunction] = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        embedding_size: int | None = None,
        bias: bool = False,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = True,
        query_length: int | None = None,
        document_length: int | None = None,
        attend_to_expansion_tokens: bool | None = None,
        skiplist_words: list[str] | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
        model_card_data: PylateModelCardData | None = None,
    ) -> None:
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_length = query_length
        self.document_length = document_length
        self.attend_to_expansion_tokens = attend_to_expansion_tokens
        self.skiplist_words = skiplist_words
        model_card_data = model_card_data or PylateModelCardData()
        if similarity_fn_name is None:
            similarity_fn_name = "MaxSim"

        super(ColBERT, self).__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            similarity_fn_name=similarity_fn_name,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            use_auth_token=use_auth_token,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
        )
        hidden_size = self[0].get_word_embedding_dimension()

        # Add a linear projection layer to the model in order to project the embeddings to the desired size.
        if len(self) < 2:
            # If the model is a stanford-nlp ColBERT, load the weights of the dense layer
            if (
                self[0].auto_model.config.architectures is not None
                and self[0].auto_model.config.architectures[0] == "HF_ColBERT"
            ):
                self.append(
                    Dense.from_stanford_weights(
                        model_name_or_path,
                        cache_folder,
                        revision,
                        local_files_only,
                        token,
                        use_auth_token,
                    )
                )
                logger.info("Loaded the weights from Stanford NLP model.")
                try:
                    metadata = cached_file(
                        model_name_or_path,
                        filename="artifact.metadata",
                        cache_dir=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                        token=token,
                        use_auth_token=use_auth_token,
                    )
                    with open(metadata, "r") as f:
                        metadata = json.load(f)
                        # If the user do not override the values, read from config file
                        if self.query_prefix is None:
                            self.query_prefix = metadata["query_token_id"]
                        if self.document_prefix is None:
                            self.document_prefix = metadata["doc_token_id"]
                        if self.query_length is None:
                            self.query_length = metadata["query_maxlen"]
                        if self.document_length is None:
                            self.document_length = metadata["doc_maxlen"]
                        if self.attend_to_expansion_tokens is None:
                            self.attend_to_expansion_tokens = metadata[
                                "attend_to_mask_tokens"
                            ]
                    logger.info("Loaded the configuration from Stanford NLP model.")
                except EnvironmentError:
                    if self.query_prefix is None:
                        self.query_prefix = "[unused0]"
                    if self.document_prefix is None:
                        self.document_prefix = "[unused1]"
                    # We do not set the query/doc length as they'll be set to the default values afterwards. We do it for prefixes as the default from stanford is different from ours
                    logger.warning(
                        "Could not load the configuration file from Stanford NLP model, using default values."
                    )
            else:
                # Add a linear projection layer to the model in order to project the embeddings to the desired size
                embedding_size = embedding_size or 128

                logger.info(
                    f"The checkpoint does not contain a linear projection layer. Adding one with output dimensions ({hidden_size}, {embedding_size})."
                )
                logger.info("Created a PyLate model from base encoder.")
                self.append(
                    Dense(
                        in_features=hidden_size, out_features=embedding_size, bias=bias
                    )
                )

        elif (
            embedding_size is not None
            and self[1].get_sentence_embedding_dimension() != embedding_size
        ):
            logger.warning(
                f"The checkpoint contains a dense layer with output dimension ({hidden_size}, {self[1].get_sentence_embedding_dimension()}). Replacing it with a Dense layer with output dimensions ({hidden_size}, {embedding_size})."
            )
            self[1] = Dense(
                in_features=hidden_size, out_features=embedding_size, bias=bias
            )

        elif not isinstance(self[1], Dense):
            logger.warning(
                f"Converting the existing Dense layer from SentenceTransform with output dimensions ({hidden_size}, {self[1].get_sentence_embedding_dimension()})."
            )
            self[1] = Dense.from_sentence_transformers(dense=self[1])
        else:
            logger.info("PyLate model loaded successfully.")

        # Ensure all tensors in the model are of the same dtype as the first tensor
        try:
            dtype = next(self.parameters()).dtype
            self.to(dtype)
        except StopIteration:
            pass

        self.to(device)
        self.is_hpu_graph_enabled = False
        # Override the configuration values with the provided arguments, if any. If not set and values have not been read from configs, set to default values.
        self.query_prefix = (
            query_prefix if query_prefix is not None else self.query_prefix or "[Q] "
        )
        self.document_prefix = (
            document_prefix
            if document_prefix is not None
            else self.document_prefix or "[D] "
        )

        # Try adding the prefixes to the tokenizer. We call resize_token_embeddings twice to ensure the tokens are added only if resize_token_embeddings works. There should be a better way to do this.
        try:
            self._first_module().auto_model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.add_tokens([self.query_prefix, self.document_prefix])
            self._first_module().auto_model.resize_token_embeddings(len(self.tokenizer))
        except NotImplementedError:
            logger.warning(
                "The tokenizer does not support resizing the token embeddings, the prefixes token have not been added to vocabulary."
            )

        self.document_prefix_id = self.tokenizer.convert_tokens_to_ids(
            self.document_prefix
        )

        # Set the query prefix ID using the tokenizer.
        self.query_prefix_id = self.tokenizer.convert_tokens_to_ids(self.query_prefix)

        # Set the padding token ID to be the same as the mask token ID for queries.
        self.tokenizer.pad_token_id = self.tokenizer.mask_token_id

        self.document_length = (
            document_length
            if document_length is not None
            else self.document_length or 180
        )

        self.query_length = (
            query_length if query_length is not None else self.query_length or 32
        )

        self.skiplist_words = (
            skiplist_words
            if skiplist_words is not None
            else self.skiplist_words or list(string.punctuation)
        )

        # Convert skiplist words to their corresponding token IDs.
        self.skiplist = [
            self.tokenizer.convert_tokens_to_ids(word) for word in self.skiplist_words
        ]
        self.attend_to_expansion_tokens = (
            attend_to_expansion_tokens
            if attend_to_expansion_tokens is not None
            else self.attend_to_expansion_tokens or False
        )

    @staticmethod
    def load(input_path) -> "ColBERT":
        return ColBERT(model_name_or_path=input_path)

    def __len__(self) -> int:
        return len(self._modules)

    @staticmethod
    def insert_prefix_token(input_ids: torch.Tensor, prefix_id: int) -> torch.Tensor:
        """Inserts a prefix token at the beginning of each sequence in the input tensor."""
        prefix_tensor = torch.full(
            size=(input_ids.size(dim=0), 1),
            fill_value=prefix_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        return torch.cat(
            tensors=[input_ids[:, :1], prefix_tensor, input_ids[:, 1:]], dim=1
        )

    def encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        padding: bool = False,
        device: str = None,
        normalize_embeddings: bool = True,
        is_query: bool = True,
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[torch.Tensor] | ndarray | torch.Tensor:
        """
        Computes sentence embeddings.

        Parameters
        ----------
        sentences
            The sentences to embed.
        prompt_name
            The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary, which is either set in
            the constructor or loaded from the model configuration. For example, if `prompt_name` is "query" and the
            `prompts` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as
            "query: What is the capital of France?" because the sentence is appended to the prompt. If `prompt` is also
            set, this argument is ignored. Defaults to None.
        prompt
            The prompt to use for encoding. For example, if the prompt is "query: ", then the sentence "What is the capital
            of France?" will be encoded as "query: What is the capital of France?" because the sentence is appended to the
            prompt. If `prompt` is set, `prompt_name` is ignored. Defaults to None.
        batch_size
            The batch size used for the computation. Defaults to 32.
        show_progress_bar
            Whether to output a progress bar when encoding sentences. Defaults to None.
        output_value
            The type of embeddings to return: "sentence_embedding" to get sentence embeddings, "token_embeddings" to get
            wordpiece token embeddings, and `None` to get all output values. Defaults to "sentence_embedding".
        precision
            The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". All
            non-float32 precisions are quantized embeddings. Quantized embeddings are smaller in size and faster to compute,
            but may have lower accuracy. They are useful for reducing the size of the embeddings of a corpus for semantic
            search, among other tasks. Defaults to "float32".
        convert_to_numpy
            Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors. Defaults to True.
        convert_to_tensor
            Whether the output should be one large tensor. Overwrites `convert_to_numpy`. Defaults to False.
        device
            Which :class:`torch.device` to use for the computation. Defaults to None.
        normalize_embeddings
            Whether to normalize returned vectors to have length 1. In that case, the faster dot-product (util.dot_score)
            instead of cosine similarity can be used. Defaults to False.
        is_query
            Whether the input sentences are queries. If True, the query prefix is added to the input sentences and the
            sequence is padded; otherwise, the document prefix is added and the sequence is not padded. Defaults to True.
        pool_factor
            The factor by which to pool the document embeddings, resulting in 1/pool_factor of the original tokens. If set
            to 1, no pooling is done; if set to 2, 50% of the tokens are kept; if set to 3, 33%, and so on. Defaults to 1.
        protected_tokens
            The number of tokens at the beginning of the sequence that should not be pooled. Defaults to 1 (CLS token).

        """
        if isinstance(sentences, list):
            # If we have a list of list of sentences, we encode each list separately.
            if isinstance(sentences[0], list):
                embeddings = []

                for batch in sentences:
                    batch_embedings = self.encode(
                        sentences=batch,
                        prompt_name=prompt_name,
                        prompt=prompt,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                        precision=precision,
                        convert_to_numpy=convert_to_numpy,
                        convert_to_tensor=convert_to_tensor,
                        padding=padding,
                        device=device,
                        normalize_embeddings=normalize_embeddings,
                        is_query=is_query,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                    batch_embedings = (
                        torch.stack(batch_embedings)
                        if convert_to_tensor
                        else batch_embedings
                    )

                    embeddings.append(batch_embedings)

                return embeddings

        if self.device.type == "hpu" and not self.is_hpu_graph_enabled:
            import habana_frameworks.torch as ht

            ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
            self.is_hpu_graph_enabled = True

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        # Convert to tensor takes precedence over convert to numpy
        if not convert_to_numpy:
            convert_to_tensor = True
        convert_to_numpy = not convert_to_tensor

        # TODO: We cannot convert to tensor/numpy for token embeddings as they are not the same size
        # if output_value != "sentence_embedding":
        # convert_to_tensor = False
        # convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if prompt is not None and prompt_name is not None:
            logger.warning(
                "Provide either a `prompt` or a `prompt_name`, not both. "
                "Ignoring the `prompt_name` in favor of the provided `prompt`."
            )

        elif prompt is None:
            if prompt_name is not None:
                prompt = self.prompts.get(prompt_name)
                if prompt is None:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary. "
                        f"Available keys are: {list(self.prompts.keys())!r}."
                    )
            else:
                prompt = self.prompts.get(self.default_prompt_name)

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models require removing the prompt before pooling (e.g. Instructor, Grit).
            # Tracking the prompt length allow us to remove the prompt during pooling.
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = (
                    tokenized_prompt["input_ids"].shape[-1] - 1
                )

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc=f"Encoding queries (bs={batch_size})"
            if is_query
            else f"Encoding documents (bs={batch_size})",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(texts=sentences_batch, is_query=is_query)

            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape

                    additional_pad_len = (
                        2 ** math.ceil(math.log2(curr_tokenize_len[1]))
                        - curr_tokenize_len[1]
                    )

                    features["input_ids"] = torch.cat(
                        tensors=(
                            features["input_ids"],
                            torch.ones(
                                size=(curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        dim=-1,
                    )

                    features["attention_mask"] = torch.cat(
                        tensors=(
                            features["attention_mask"],
                            torch.zeros(
                                size=(curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        dim=-1,
                    )

                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            tensors=(
                                features["token_type_ids"],
                                torch.zeros(
                                    size=(curr_tokenize_len[0], additional_pad_len),
                                    dtype=torch.int8,
                                ),
                            ),
                            dim=-1,
                        )

            features = batch_to_device(batch=features, target_device=device)
            features.update(extra_features)

            with torch.no_grad():
                # TODO: add the truncate/sliding window logic here
                out_features = self.forward(input=features)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                if not is_query:
                    # Compute the mask for the skiplist (punctuation symbols)
                    skiplist_mask = self.skiplist_mask(
                        input_ids=features["input_ids"], skiplist=self.skiplist
                    )
                    masks = torch.logical_and(
                        input=skiplist_mask, other=out_features["attention_mask"]
                    )
                else:
                    # We keep all tokens in the query (no skiplist) and we do not want to prune expansion tokens in queries even if we do not attend to them in attention layers
                    masks = torch.ones_like(
                        input=out_features["input_ids"], dtype=torch.bool
                    )

                embeddings = []
                for (
                    token_embedding,
                    mask,
                ) in zip(out_features["token_embeddings"], masks):
                    token_embedding = (
                        torch.nn.functional.normalize(
                            input=token_embedding[mask], p=2, dim=1
                        )
                        if normalize_embeddings
                        else token_embedding[mask]
                    )
                    embeddings.append(token_embedding)

                # Pool factor must be greater than 1: keeping 1 over pool_factor tokens embeddings.
                if pool_factor > 1 and not is_query:
                    embeddings = self.pool_embeddings_hierarchical(
                        documents_embeddings=embeddings,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = [embedding.cpu() for embedding in embeddings]

                all_embeddings.extend(embeddings)

        # Pad the embeddings to the same length. Documents can have different lengths while queries are already padded.
        if padding:
            all_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=all_embeddings, batch_first=True, padding_value=0
            )

            # Create a list of tensors.
            all_embeddings = torch.split(
                tensor=all_embeddings, split_size_or_sections=1, dim=0
            )

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(
                embeddings=all_embeddings, precision=precision
            )

        # Return a list of arrays instead of single contiguous array since documents can have different lengths.
        if convert_to_tensor:
            if not len(all_embeddings):
                return torch.tensor()

            if isinstance(all_embeddings, np.ndarray):
                all_embeddings = [
                    torch.from_numpy(ndarray=embedding) for embedding in all_embeddings
                ]

        elif convert_to_numpy:
            bloat = all_embeddings[0].dtype == torch.bfloat16
            all_embeddings = [
                embedding.float().numpy() if bloat else embedding.numpy()
                for embedding in all_embeddings
            ]

        return all_embeddings[0] if input_was_string else all_embeddings

    def pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[torch.Tensor]:
        """
        Pools the embeddings hierarchically by clustering and averaging them.

        Parameters
        ----------
        document_embeddings_list
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters. Defaults to 1.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document. Defaults to 1.

        Returns
        -------
            A list of pooled embeddings for each document.
        """
        device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        pooled_embeddings = []

        for document_embeddings in documents_embeddings:
            document_embeddings = document_embeddings.to(device=device)

            # Separate protected tokens from the rest
            protected_embeddings = document_embeddings[:protected_tokens]
            embeddings_to_pool = document_embeddings[protected_tokens:]

            # Compute cosine similarity and convert to distance matrix
            cosine_similarities = torch.mm(
                input=embeddings_to_pool, mat2=embeddings_to_pool.t()
            )
            distance_matrix = 1 - cosine_similarities.cpu().numpy()

            # Perform hierarchical clustering using Ward's method
            clusters = hierarchy.linkage(distance_matrix, method="ward")
            num_embeddings = len(embeddings_to_pool)

            # Determine the number of clusters based on pool_factor
            num_clusters = max(num_embeddings // pool_factor, 1)
            cluster_labels = hierarchy.fcluster(
                clusters, t=num_clusters, criterion="maxclust"
            )

            # Pool embeddings within each cluster
            pooled_document_embeddings = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_indices = torch.where(
                    condition=torch.tensor(
                        data=cluster_labels == cluster_id, device=device
                    )
                )[0]
                if cluster_indices.numel() > 0:
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
                    pooled_document_embeddings.append(cluster_embedding)

            # Re-append protected embeddings
            pooled_document_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))

        return pooled_embeddings

    @property
    def similarity_fn_name(self) -> Literal["MaxSim"]:
        """Return the name of the similarity function used by :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise`.

        Returns:
            Optional[str]: The name of the similarity function. Can be None if not set, in which case it will
                default to "cosine" when first called.
        Examples
        --------
        >>> model = ColBERT("bert-base-uncased")
        >>> model.similarity_fn_name
            'MaxSim'
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.MAXSIM
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(self, value: Literal["MaxSim"] | SimilarityFunction) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(
                value
            )

    @staticmethod
    def skiplist_mask(input_ids: torch.Tensor, skiplist: list[int]) -> torch.Tensor:
        """Create a mask for the set of input_ids that are in the skiplist."""
        skiplist = torch.tensor(
            data=skiplist, dtype=torch.long, device=input_ids.device
        )

        # Create a tensor of ones with the same shape as input_ids.
        mask = torch.ones_like(input=input_ids, dtype=torch.bool)

        # Update the mask for each token in the skiplist.
        for token_id in skiplist:
            mask = torch.where(
                condition=input_ids == token_id,
                input=torch.tensor(data=0, dtype=torch.bool, device=input_ids.device),
                other=mask,
            )

        return mask

    def start_multi_process_pool(self, target_devices: list[str] = None) -> dict:
        """Starts a multi-process pool to process the encoding with several independent processes.
        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Parameters
        ----------
        target_devices
            PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...], ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"].
            If target_devices is None and CUDA/NPU is available, then all available CUDA/NPU devices will be used. If target_devices is None and
            CUDA/NPU is not available, then 4 CPU devices will be used.

        Returns
        -------
            A dictionary with the target processes, an input queue, and an output queue.
        """
        return _start_multi_process_pool(model=self, target_devices=target_devices)

    def encode_multi_process(
        self,
        sentences: list[str],
        pool: dict[str, object],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        chunk_size: int = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        normalize_embeddings: bool = True,
        padding: bool = False,
        is_query: bool = True,
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[np.ndarray]:
        """
        Encodes a list of sentences using multiple processes and GPUs via
        :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.
        The sentences are chunked into smaller packages and sent to individual processes, which encode them on different
        GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

        Parameters
        ----------
        sentences
            List of sentences to encode.
        pool
            A pool of workers started with SentenceTransformer.start_multi_process_pool.
        prompt_name
            The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
            which is either set in the constructor or loaded from the model configuration. For example if
            ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
            is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
            is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
        prompt
            The prompt to use for encoding. For example, if the prompt is "query: ", then the
            sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
            because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
        batch_size
            Encode sentences with batch size. (default: 32)
        chunk_size
            Sentences are chunked and sent to the individual processes. If None, it determines a
            sensible size. Defaults to None.
        precision
            The precision to use for the
            embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions
            are quantized embeddings. Quantized embeddings are smaller in size and faster to compute, but may
            have lower accuracy. They are useful for reducing the size of the embeddings of a corpus for
            semantic search, among other tasks. Defaults to "float32".
        normalize_embeddings
            Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to True.

        Examples
        --------
        >>> from pylate import models

        >>> model = models.ColBERT(
        ...     "sentence-transformers/all-MiniLM-L6-v2",
        ...     device="cpu"
        ... )

        >>> pool = model.start_multi_process_pool()
        >>> embeddings = model.encode_multi_process(
        ...     sentences=["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."],
        ...     pool=pool,
        ...     batch_size=1,
        ...     is_query=True,
        ... )
        >>> model.stop_multi_process_pool(pool)

        >>> assert len(embeddings) == 3

        """

        if chunk_size is None:
            chunk_size = min(
                math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000
            )

        logger.debug(
            f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}"
        )

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [
                        last_chunk_id,
                        batch_size,
                        chunk,
                        prompt_name,
                        prompt,
                        precision,
                        normalize_embeddings,
                        padding,
                        is_query,
                        pool_factor,
                        protected_tokens,
                    ]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put(
                [
                    last_chunk_id,
                    batch_size,
                    chunk,
                    prompt_name,
                    prompt,
                    precision,
                    normalize_embeddings,
                    padding,
                    is_query,
                    pool_factor,
                    protected_tokens,
                ]
            )
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0]
        )
        return [np.concatenate(result[1]) for result in results_list]

    def tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        is_query: bool = True,
        pad_document: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenizes the input texts.

        Args:
            texts (Union[list[str], list[dict], list[tuple[str, str]]]): A list of texts to be tokenized.
            is_query (bool): Flag to indicate if the texts are queries. Defaults to True.
            pad_document (bool): Flag to indicate if documents should be padded to max length. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors with the tokenized texts, including "input_ids",
                "attention_mask", and optionally "token_type_ids".
        """
        # Set max sequence length based on whether the input is a query or document
        max_length = self.query_length if is_query else self.document_length
        self._first_module().max_seq_length = (
            max_length - 1
        )  # Subtract 1 for the prefix token

        # Pad queries (query expansion) and handle padding for documents if specified
        tokenize_args = {"padding": "max_length"} if pad_document or is_query else {}

        # Tokenize the texts
        tokenized_outputs = self._first_module().tokenize(texts, **tokenize_args)

        # Determine prefix ID based on input type
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id

        # Insert prefix token and update attention mask
        tokenized_outputs["input_ids"] = self.insert_prefix_token(
            tokenized_outputs["input_ids"], prefix_id
        )
        tokenized_outputs["attention_mask"] = self.insert_prefix_token(
            tokenized_outputs["attention_mask"], 1
        )

        # Update token type IDs if they exist
        if "token_type_ids" in tokenized_outputs:
            tokenized_outputs["token_type_ids"] = self.insert_prefix_token(
                tokenized_outputs["token_type_ids"], 0
            )

        # Adjust attention mask for expansion tokens if required
        if is_query and self.attend_to_expansion_tokens:
            tokenized_outputs["attention_mask"].fill_(1)

        return tokenized_outputs

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SentenceTransformer(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (list[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        super().save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            config = self._model_config.copy()
            config["prompts"] = self.prompts
            config["default_prompt_name"] = self.default_prompt_name
            config["similarity_fn_name"] = self.similarity_fn_name
            config["query_prefix"] = self.query_prefix
            config["document_prefix"] = self.document_prefix
            config["query_length"] = self.query_length
            config["document_length"] = self.document_length
            config["attend_to_expansion_tokens"] = self.attend_to_expansion_tokens
            config["skiplist_words"] = self.skiplist_words
            json.dump(config, fOut, indent=2)

    def _load_auto_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
    ) -> list[nn.Module]:
        """Create a Transformer model from a model name or path. This module is distinct
        from SentenceTransformer as it do not set the pooling layer.

        Parameters
        ----------
        model_name_or_path
            The name or path of the pre-trained model.
        token
            The token to use for the model.
        cache_folder
            The folder to cache the model.
        revision
            The revision of the model. Defaults to None.
        trust_remote_code
            Whether to trust remote code. Defaults to False.
        local_files_only
            Whether to use only local files. Defaults to False.
        model_kwargs
            Additional keyword arguments for the model. Defaults to None.
        tokenizer_kwargs
            Additional keyword arguments for the tokenizer. Defaults to None.
        config_kwargs
            Additional keyword arguments for the config. Defaults to None.

        """
        logger.warning(
            f"No sentence-transformers model found with name {model_name_or_path}."
        )

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }

        model_kwargs = (
            shared_kwargs if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        )

        tokenizer_kwargs = (
            shared_kwargs
            if tokenizer_kwargs is None
            else {**shared_kwargs, **tokenizer_kwargs}
        )

        config_kwargs = (
            shared_kwargs
            if config_kwargs is None
            else {**shared_kwargs, **config_kwargs}
        )

        transformer_model = Transformer(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
        )

        self.model_card_data.set_base_model(
            model_id=model_name_or_path, revision=revision
        )

        return [transformer_model]

    def _load_sbert_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
    ) -> list[nn.Module]:
        """Create a Sentence Transformer model from a model name or path."""
        modules, module_kwargs = super()._load_sbert_model(
            model_name_or_path=model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
        )

        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path=model_name_or_path,
            filename="config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        if config_sentence_transformers_json_path is not None:
            with open(file=config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fp=fIn)

            # Loading the query/document prefixes and query_length
            if "query_prefix" in self._model_config:
                self.query_prefix = self._model_config["query_prefix"]
            if "document_prefix" in self._model_config:
                self.document_prefix = self._model_config["document_prefix"]
            if "query_length" in self._model_config:
                self.query_length = self._model_config["query_length"]
            if "document_length" in self._model_config:
                self.document_length = self._model_config["document_length"]
            if "attend_to_expansion_tokens" in self._model_config:
                self.attend_to_expansion_tokens = self._model_config[
                    "attend_to_expansion_tokens"
                ]
            if "skiplist_words" in self._model_config:
                self.skiplist_words = self._model_config["skiplist_words"]

        return [
            module
            for module in modules.values()
            if isinstance(module, Transformer)
            or isinstance(module, DenseSentenceTransformer)
        ], module_kwargs
