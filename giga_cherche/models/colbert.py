import copy
import importlib
import json
import logging
import math
import os
import queue
import string
import tempfile
import traceback
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Literal,
    Optional,
    Union,
    overload,
)

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import HfApi
from numpy import ndarray
from scipy.cluster.hierarchy import fcluster, linkage
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.fit_mixin import FitMixin
from sentence_transformers.model_card import (
    SentenceTransformerModelCardData,
    generate_model_card,
)
from sentence_transformers.models import Normalize, Pooling, Transformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import (
    batch_to_device,
    get_device_name,
    import_from_string,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
)
from torch import nn
from tqdm.autonotebook import trange
from transformers import is_torch_npu_available

from ..utils import HUGGINGFACE_MODELS
from .LinearProjection import LinearProjection

logger = logging.getLogger(__name__)


__version__ = "3.0.0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"


class ColBERT(nn.Sequential, FitMixin):
    """
    Loads or creates a ColBERT model that can be used to map sentences / text to multi-vectors embeddings.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SentenceTransformer models from scratch.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        prompts (dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text.
            The prompt text will be prepended before any text to encode. For example:
            `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
            titles in "}`.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use. Valid options are "cosine", "dot",
            "euclidean", and "manhattan". If not set, it is automatically set to "cosine" if `similarity` or
            `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.
        cache_folder (str, optional): Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
        use_auth_token (bool or str, optional): Deprecated argument. Please use `token` instead.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is
            only applicable during inference when :meth:`SentenceTransformer.encode` is called.
        model_kwargs (dict, optional): Additional model configuration parameters to be passed to the Huggingface Transformers model.
            Particularly useful options are:

            - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`.
              The different options are:

                    1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified
                    ``dtype``, ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will
                    get loaded in ``torch.float`` (fp32).

                    2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be
                    attempted to be used. If this entry isn't found then next check the ``dtype`` of the first weight in
                    the checkpoint that's of a floating point type and use that as ``dtype``. This will load the model
                    using the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how
                    the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
            - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
              `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
              <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
              or `"flash_attention_2"` (using `Dao-AILab/flash-attention <https://github.com/Dao-AILab/flash-attention>`_).
              By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
              implementation.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (dict, optional): Additional tokenizer configuration parameters to be passed to the Huggingface Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (dict, optional): Additional model configuration parameters to be passed to the Huggingface Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details.
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
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
            The length of the query to truncate / pad to with mask tokens. If set, will override the config value. Default to 32.
        document_length
            The max length of the document to truncate. If set, will override the config value. Default to 180.
        attend_to_expansion_tokens
            Whether to attend to the expansion tokens in the attention layers model. If False, the original tokens will not only attend to the expansion tokens, only the expansion tokens will attend to the original tokens. (Default was False in original ColBERT codebase)

    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[1.0000, 0.6817, 0.0492],
            #         [0.6817, 1.0000, 0.0421],
            #         [0.0492, 0.0421, 1.0000]])
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
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
        model_card_data: Optional[SentenceTransformerModelCardData] = None,
        embedding_size: int | None = 128,
        query_prefix: str | None = "[Q] ",
        document_prefix: str | None = "[D] ",
        add_special_tokens: bool = True,
        truncation: bool = True,
        query_length: int | None = None,
        document_length: int | None = None,
        attend_to_expansion_tokens: bool = False,
        skiplist_words: list[str] | None = None,
    ):
        # Note: self._load_sbert_model can also update `self.prompts` and `self.default_prompt_name`
        self.prompts = prompts or {}
        self.default_prompt_name = default_prompt_name
        self.similarity_fn_name = similarity_fn_name
        self.truncate_dim = truncate_dim
        self.model_card_data = model_card_data or SentenceTransformerModelCardData()
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_length = query_length
        self.document_length = document_length
        self.attend_to_expansion_tokens = attend_to_expansion_tokens
        self.skiplist_words = skiplist_words
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v3 of SentenceTransformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")

        if device is None:
            device = get_device_name()
            logger.info("Use pytorch device_name: {}".format(device))

        if device == "hpu" and importlib.util.find_spec("optimum") is not None:
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(
                "Load pretrained SentenceTransformer: {}".format(model_name_or_path)
            )

            if not os.path.exists(model_name_or_path):
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if (
                    "/" not in model_name_or_path
                    and model_name_or_path.lower() not in HUGGINGFACE_MODELS
                ):
                    # A model from sentence-transformers
                    model_name_or_path = (
                        __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path
                    )
            # This is actually not required as even if we pool, we also return the unpooled embeddings and are using these
            if model_kwargs is None:
                model_kwargs = {}
            model_kwargs["add_pooling_layer"] = False

            if is_sentence_transformer_model(
                model_name_or_path,
                token,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
            ):
                modules = self._load_sbert_model(
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
            else:
                modules = self._load_auto_model(
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

        # modules = [module for module in modules if isinstance(module, Transformer) or isinstance(module, LinearProjection)]
        # print(modules)
        hidden_size = modules[0].get_word_embedding_dimension()
        # If there is no linear projection layer, add one
        # TODO: do this more cleanly
        if len(modules) < 2:
            logger.warning(
                f"The checkpoint does not contain a linear projection layer. Adding one with output dimensions ({hidden_size}, {embedding_size})"
            )
            # Linear layer from hidden_size to target embedding_size, original ColBERT does not have bias in the linear layer
            linear_layer = LinearProjection(hidden_size, embedding_size, bias=False)
            # modules.append(nn.Linear(hidden_size, embedding_size, bias=False))
            modules.append(linear_layer)
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)]
            )

        super().__init__(modules)

        self.to(device)
        self.is_hpu_graph_enabled = False

        if (
            self.default_prompt_name is not None
            and self.default_prompt_name not in self.prompts
        ):
            raise ValueError(
                f"Default prompt name '{self.default_prompt_name}' not found in the configured prompts "
                f"dictionary with keys {list(self.prompts.keys())!r}."
            )

        if self.prompts:
            logger.info(
                f"{len(self.prompts)} prompts are loaded, with the keys: {list(self.prompts.keys())}"
            )
        if self.default_prompt_name:
            logger.warning(
                f"Default prompt name is set to '{self.default_prompt_name}'. "
                "This prompt will be applied to all `encode()` calls, except if `encode()` "
                "is called with `prompt` or `prompt_name` parameters."
            )

        # Ideally, INSTRUCTOR models should set `include_prompt=False` in their pooling configuration, but
        # that would be a breaking change for users currently using the InstructorEmbedding project.
        # So, instead we hardcode setting it for the main INSTRUCTOR models, and otherwise give a warning if we
        # suspect the user is using an INSTRUCTOR model.
        if model_name_or_path in (
            "hkunlp/instructor-base",
            "hkunlp/instructor-large",
            "hkunlp/instructor-xl",
        ):
            self.set_pooling_include_prompt(include_prompt=False)
        elif (
            model_name_or_path
            and "/" in model_name_or_path
            and "instructor" in model_name_or_path.split("/")[1].lower()
        ):
            if any(
                [
                    module.include_prompt
                    for module in self
                    if isinstance(module, Pooling)
                ]
            ):
                logger.warning(
                    "Instructor models require `include_prompt=False` in the pooling configuration. "
                    "Either update the model configuration or call `model.set_pooling_include_prompt(False)` after loading the model."
                )

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)

        # this will add the query and document prefix to the tokenizer vocab if they are not already there and resize the embeddings accordingly
        self.tokenizer.add_tokens([self.query_prefix, self.document_prefix])
        self._first_module().auto_model.resize_token_embeddings(len(self.tokenizer))

        self.document_prefix_id = self.tokenizer.convert_tokens_to_ids(
            self.document_prefix
        )
        self.query_prefix_id = self.tokenizer.convert_tokens_to_ids(self.query_prefix)
        # We are using the mask token as the padding token for padding queries
        self.tokenizer.pad_token_id = self.tokenizer.mask_token_id

        # We override the config values with the ones provided by the user
        if document_length is not None:
            self.document_length = document_length
        if query_length is not None:
            self.query_length = query_length
        if skiplist_words is not None:
            self.skiplist_words = skiplist_words
        # If no values are provided and there is no value in the config, use default values
        if self.document_length is None:
            self.document_length = 180
        if self.query_length is None:
            self.query_length = 32
        if self.skiplist_words is None:
            self.skiplist_words = [symbol for symbol in string.punctuation]
        # Converting to ids (we do not store the ids in the config because it is less readable)
        self.skiplist = [
            self.tokenizer.convert_tokens_to_ids(word) for word in self.skiplist_words
        ]

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
        normalize_embeddings: bool = True,  # Normalize the embedding to compute cosine similarity
        is_query: bool = True,
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[torch.Tensor] | ndarray | torch.Tensor:
        """
        Computes sentence embeddings.

        Args:
            sentences (str | list[str]): The sentences to embed.
            prompt_name (str | None, optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (str | None, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            is_query (bool, optional): Whether the input sentences are queries. If True, the query prefix is added to the input sentences and the sequence is padded, else the document prefix is added and the sequence is not padded. Defaults to True.
            pool_factor (int, optional): The factor by which to pool the documents embeddings, resulting in 1/pool_factor of the original token. If set to 1, no pooling is done, 2 keep only 50% of the tokens, 3, 33%, ... Defaults to 1.
            protected_tokens (int, optional): The number of tokens at the beginning of the sequence that should not be pooled. Defaults to 1 (CLS token).

        Returns:
            list[torch.Tensor] | ndarray | torch.Tensor: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch torch.Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                # Load a pre-trained SentenceTransformer model
                model = SentenceTransformer('all-mpnet-base-v2')

                # Encode some texts
                sentences = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 768)
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

        if convert_to_tensor:
            convert_to_numpy = False

        # TODO: We cannot convert to tensor/numpy for token embeddings as they are not the same size
        # if output_value != "sentence_embedding":
        # convert_to_tensor = False
        # convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
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
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, is_query=is_query)

            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape
                    additional_pad_len = (
                        2 ** math.ceil(math.log2(curr_tokenize_len[1]))
                        - curr_tokenize_len[1]
                    )
                    features["input_ids"] = torch.cat(
                        (
                            features["input_ids"],
                            torch.ones(
                                (curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        -1,
                    )
                    features["attention_mask"] = torch.cat(
                        (
                            features["attention_mask"],
                            torch.zeros(
                                (curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        -1,
                    )
                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            (
                                features["token_type_ids"],
                                torch.zeros(
                                    (curr_tokenize_len[0], additional_pad_len),
                                    dtype=torch.int8,
                                ),
                            ),
                            -1,
                        )

            features = batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                # TODO: add the truncate/sliding window logic here
                out_features = self.forward(features)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                # out_features["sentence_embedding"] = truncate_embeddings(
                #     out_features["sentence_embedding"], self.truncate_dim
                # )
                if not is_query:
                    # Compute the mask for the skiplist (punctuation symbols)
                    skiplist_mask = self.skiplist_mask(
                        features["input_ids"], skiplist=self.skiplist
                    )
                    masks = torch.logical_and(
                        skiplist_mask, out_features["attention_mask"]
                    )
                else:
                    # We keep all tokens in the query (no skiplist) and we do not want to prune expansion tokens in queries even if we do not attend to them in attention layers
                    masks = torch.ones_like(out_features["input_ids"], dtype=torch.bool)

                embeddings = []
                for (
                    token_emb,
                    mask,
                ) in zip(out_features["token_embeddings"], masks):
                    token_emb = token_emb[mask]
                    # TODO: normalize at the list level/use the module Normalize?
                    if normalize_embeddings:
                        token_emb = torch.nn.functional.normalize(token_emb, p=2, dim=1)
                    embeddings.append(token_emb)
                # If we are encoding documents and the pool factor is greater than 1, we pool the embeddings
                if pool_factor > 1 and not is_query:
                    embeddings = self.pool_embeddings_hierarchical(
                        embeddings, pool_factor, protected_tokens
                    )
                # elif output_value is None:  # Return all outputs
                #     embeddings = []
                #     for sent_idx in range(len(out_features["sentence_embedding"])):
                #         row = {name: out_features[name][sent_idx] for name in out_features}
                #         embeddings.append(row)
                # else:  # Sentence embeddings
                #     embeddings = out_features[output_value]
                #     embeddings = embeddings.detach()
                # if normalize_embeddings:
                #     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    # TODO: convert to tensor to do all .cpu() at once?
                    embeddings = [emb.cpu() for emb in embeddings]
                    # embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)
        # TODO: right now we are casting the embeddings to a list of tensors for consistency, but this results in an useless overhead to convert them back to a tensor afterwards.
        if padding:
            # This will pad the documents to the same length, queries are already the same lengths
            all_embeddings = torch.nn.utils.rnn.pad_sequence(
                all_embeddings, batch_first=True, padding_value=0
            )
            # We create a list of tensor from the big padded tensor
            all_embeddings = torch.split(all_embeddings, 1, dim=0)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                # We return a list of tensors and not a big tensor because we cannot guarantee all element have the same sequence length
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = [
                        torch.from_numpy(embedding) for embedding in all_embeddings
                    ]
                # Else, we already have a list of tensors, the expected output
            else:
                all_embeddings = torch.tensor()
        elif convert_to_numpy:
            # We return a list of numpy arrays and not a big numpy array because we cannot guarantee all element have the same sequence length
            if all_embeddings[0].dtype == torch.bfloat16:
                all_embeddings = [emb.float().numpy() for emb in all_embeddings]
            else:
                all_embeddings = [emb.numpy() for emb in all_embeddings]

        return all_embeddings[0] if input_was_string else all_embeddings

    # TODO: add typing
    """
        Computes sentence embeddings.

        Args:
            documents_embeddings (list[torch.Tensor]): The embeddings of the documents to pool.
            pool_factor (int, optional): The factor by which to pool the documents embeddings, resulting in 1/pool_factor of the original token. If set to 1, no pooling is done, 2 keep only 50% of the tokens, 3, 33%, ... Defaults to 1.
            protected_tokens (int, optional): The number of tokens at the beginning of the sequence that should not be pooled. Defaults to 1 (CLS token).

        Returns:
            list[torch.Tensor]: The list of pooled embeddings.
        """

    def pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pooled_embeddings = []
        for document_embeddings in documents_embeddings:
            document_pooled_embeddings = []
            document_embeddings.to(device)
            # Remove the tokens at protected_tokens indices
            protected_embeddings = document_embeddings[:protected_tokens]
            document_embeddings = document_embeddings[protected_tokens:]

            # Cosine similarity computation (vector are already normalized)
            similarities = torch.mm(document_embeddings, document_embeddings.t())

            # Convert similarities to a distance for better ward compatibility
            similarities = 1 - similarities.cpu().numpy()

            # Create hierarchical clusters using ward's method
            Z = linkage(similarities, metric="euclidean", method="ward")
            length = len(document_embeddings)
            # Determine the number of clusters we want in the end based on the pool factor
            max_clusters = length // pool_factor if length // pool_factor > 0 else 1
            cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")
            # Pool embeddings within each cluster
            for cluster_id in range(1, max_clusters + 1):
                cluster_indices = torch.where(
                    torch.tensor(cluster_labels == cluster_id, device=device)
                )[0]
                if cluster_indices.numel() > 0:
                    pooled_embedding = document_embeddings[cluster_indices].mean(dim=0)
                    document_pooled_embeddings.append(pooled_embedding)

            # Re-add the protected tokens to pooled_embeddings
            document_pooled_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(document_pooled_embeddings))

        return pooled_embeddings

    @staticmethod
    def skiplist_mask(input_ids, skiplist):
        skiplist = torch.tensor(skiplist, dtype=torch.long, device=input_ids.device)

        # Create a tensor of ones with the same shape as input_ids
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Iterate over the token_ids_to_mask and update the mask
        for token_id in skiplist:
            mask = torch.where(
                input_ids == token_id,
                torch.tensor(0, dtype=torch.bool, device=input_ids.device),
                mask,
            )

        return mask
        # Not working on MPS
        # Create a boolean mask indicating which tokens to mask
        mask = torch.isin(input_ids, skiplist)

        # Create the final mask tensor
        mask = torch.where(
            mask,
            torch.tensor(0, dtype=torch.long, device=input_ids.device),
            torch.tensor(1, dtype=torch.long, device=input_ids.device),
        )

        return mask

    @property
    def similarity_fn_name(self) -> str | None:
        """Return the name of the similarity function used by :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise`.

        Returns:
            str | None: The name of the similarity function. Can be None if not set, in which case any uses of
            :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise` default to "cosine".
        """
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(self, value: str | SimilarityFunction) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(
                value
            )

    @overload
    def similarity(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def similarity(
        self, embeddings1: ndarray, embeddings2: ndarray
    ) -> torch.Tensor: ...

    @property
    def similarity(
        self,
    ) -> Callable[
        [Union[torch.Tensor, ndarray], Union[torch.Tensor, ndarray]], torch.Tensor
    ]:
        """
        Compute the similarity between two collections of embeddings. The output will be a matrix with the similarity
        scores between all embeddings from the first parameter and all embeddings from the second parameter. This
        differs from `similarity_pairwise` which computes the similarity between each pair of embeddings.

        Args:
            embeddings1 (Union[torch.Tensor, ndarray]): [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[torch.Tensor, ndarray]): [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            torch.Tensor: A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity

    @overload
    def similarity_pairwise(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def similarity_pairwise(
        self, embeddings1: ndarray, embeddings2: ndarray
    ) -> torch.Tensor: ...

    @property
    def similarity_pairwise(
        self,
    ) -> Callable[
        [Union[torch.Tensor, ndarray], Union[torch.Tensor, ndarray]], torch.Tensor
    ]:
        """
        Compute the similarity between two collections of embeddings. The output will be a vector with the similarity
        scores between each pair of embeddings.

        Args:
            embeddings1 (Union[torch.Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[torch.Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            torch.Tensor: A [num_embeddings]-shaped torch tensor with pairwise similarity scores.
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_pairwise

    def start_multi_process_pool(self, target_devices: list[str] = None) -> dict:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Args:
            target_devices (list[str], optional): PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...],
                ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"]. If target_devices is None and CUDA/NPU
                is available, then all available CUDA/NPU devices will be used. If target_devices is None and
                CUDA/NPU is not available, then 4 CPU devices will be used.

        Returns:
            dict: A dictionary with the target processes, an input queue, and an output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [
                    "cuda:{}".format(i) for i in range(torch.cuda.device_count())
                ]
            elif is_torch_npu_available():
                target_devices = [
                    "npu:{}".format(i) for i in range(torch.npu.device_count())
                ]
            else:
                logger.info("CUDA/NPU is not available. Starting 4 CPU workers")
                target_devices = ["cpu"] * 4

        logger.info(
            "Start multi-process pool on devices: {}".format(
                ", ".join(map(str, target_devices))
            )
        )

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=ColBERT._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool.

        Args:
            pool (dict[str, object]): A dictionary containing the input queue, output queue, and process list.

        Returns:
            None
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def encode_multi_process(
        self,
        sentences: list[str],
        pool: dict[str, object],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        chunk_size: int = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """
        Encodes a list of sentences using multiple processes and GPUs via
        :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.
        The sentences are chunked into smaller packages and sent to individual processes, which encode them on different
        GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

        Args:
            sentences (list[str]): List of sentences to encode.
            pool (dict[str, object]): A pool of workers started with SentenceTransformer.start_multi_process_pool.
            prompt_name (str | None, optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (str | None, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int): Encode sentences with batch size. (default: 32)
            chunk_size (int): Sentences are chunked and sent to the individual processes. If None, it determines a
                sensible size. Defaults to None.
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"]): The precision to use for the
                embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions
                are quantized embeddings. Quantized embeddings are smaller in size and faster to compute, but may
                have lower accuracy. They are useful for reducing the size of the embeddings of a corpus for
                semantic search, among other tasks. Defaults to "float32".
            normalize_embeddings (bool): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

        Returns:
            np.ndarray: A 2D numpy array with shape [num_inputs, output_dimension].

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                def main():
                    model = SentenceTransformer("all-mpnet-base-v2")
                    sentences = ["The weather is so nice!", "It's so sunny outside.", "He's driving to the movie theater.", "She's going to the cinema."] * 1000

                    pool = model.start_multi_process_pool()
                    embeddings = model.encode_multi_process(sentences, pool)
                    model.stop_multi_process_pool(pool)

                    print(embeddings.shape)
                    # => (4000, 768)

                if __name__ == "__main__":
                    main()
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
                ]
            )
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0]
        )
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model, input_queue, results_queue
    ):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                (
                    chunk_id,
                    batch_size,
                    sentences,
                    prompt_name,
                    prompt,
                    precision,
                    normalize_embeddings,
                ) = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    prompt_name=prompt_name,
                    prompt=prompt,
                    device=target_device,
                    show_progress_bar=False,
                    precision=precision,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    def set_pooling_include_prompt(self, include_prompt: bool) -> None:
        """
        Sets the `include_prompt` attribute in the pooling layer in the model, if there is one.

        This is useful for INSTRUCTOR models, as the prompt should be excluded from the pooling strategy
        for these models.

        Args:
            include_prompt (bool): Whether to include the prompt in the pooling layer.

        Returns:
            None
        """
        for module in self:
            if isinstance(module, Pooling):
                module.include_prompt = include_prompt
                break

    def get_max_seq_length(self) -> int | None:
        """
        Returns the maximal sequence length that the model accepts. Longer inputs will be truncated.

        Returns:
            int | None: The maximal sequence length that the model accepts, or None if it is not defined.
        """
        if hasattr(self._first_module(), "max_seq_length"):
            return self._first_module().max_seq_length

        return None

    def insert_prefix_token(self, tensor, prefix_id):
        prefix_tensor = torch.full(
            (tensor.size(0), 1), prefix_id, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)

    def tokenize(
        self,
        texts: Union[list[str], list[dict], list[tuple[str, str]]],
        is_query: bool = True,
        pad_document: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenizes the texts.

        Args:
            texts (Union[list[str], list[dict], list[tuple[str, str]]]): A list of texts to be tokenized.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors with the tokenized texts. Common keys are "input_ids",
                "attention_mask", and "token_type_ids".
        """
        if is_query:
            # TODO: This is a hack to asymetrically set the max_seq_length for the query/document, change it once the Transformer module tokenize function expose a max_length argument
            self._first_module().max_seq_length = self.query_length
            features = self._first_module().tokenize(texts, padding="max_length")
            # Create a new tensor with the query prefix ID inserted after the first token
            features["input_ids"] = self.insert_prefix_token(
                features["input_ids"], self.query_prefix_id
            )
            # Update the attention mask to account for the new token
            features["attention_mask"] = self.insert_prefix_token(
                features["attention_mask"], 1
            )
            # In the original ColBERT, the original tokens do not attend to the expansion tokens (but the expansion tokens attend to original tokens)
            if self.attend_to_expansion_tokens:
                # Fill the attention mask with ones (we attend to "padding" tokens used for expansion)
                features["attention_mask"].fill_(1)
            return features
        else:
            self._first_module().max_seq_length = self.document_length
            extra_parameters = {}
            if pad_document:
                extra_parameters["padding"] = "max_length"
            features = self._first_module().tokenize(texts, **extra_parameters)
            # Create a new tensor with the document prefix ID inserted after the first token
            features["input_ids"] = self.insert_prefix_token(
                features["input_ids"], self.document_prefix_id
            )
            # Update the attention mask to account for the new token
            features["attention_mask"] = self.insert_prefix_token(
                features["attention_mask"], 1
            )

            return features

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.

        Returns:
            int | None: The number of dimensions in the output of `encode`. If it's not known, it's `None`.
        """
        output_dim = None
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(
                mod, "get_sentence_embedding_dimension", None
            )
            if callable(sent_embedding_dim_method):
                output_dim = sent_embedding_dim_method()
                break
        if self.truncate_dim is not None:
            # The user requested truncation. If they set it to a dim greater than output_dim,
            # no truncation will actually happen. So return output_dim insead of self.truncate_dim
            return min(output_dim or np.inf, self.truncate_dim)
        return output_dim

    @contextmanager
    def truncate_sentence_embeddings(self, truncate_dim: int | None):
        """
        In this context, :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` outputs
        sentence embeddings truncated at dimension ``truncate_dim``.

        This may be useful when you are using the same model for different applications where different dimensions
        are needed.

        Args:
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to. ``None`` does no truncation.

        """
        original_output_dim = self.truncate_dim
        try:
            self.truncate_dim = truncate_dim
            yield
        finally:
            self.truncate_dim = original_output_dim

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ):
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
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        modules_config = []

        # Save some model info
        if "__version__" not in self._model_config:
            self._model_config["__version__"] = {
                "sentence_transformers": __version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            }

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

        # Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(
                module, Transformer
            ):  # Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            if isinstance(module, Transformer):
                module.save(model_path, safe_serialization=safe_serialization)
            else:
                module.save(model_path)

            modules_config.append(
                {
                    "idx": idx,
                    "name": name,
                    "path": os.path.basename(model_path),
                    "type": type(module).__module__,
                }
            )

        with open(os.path.join(path, "modules.json"), "w") as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def save_pretrained(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ):
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
        self.save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    def _create_model_card(
        self,
        path: str,
        model_name: str | None = None,
        train_datasets: list[str] | None = "deprecated",
    ):
        """
        Create an automatic model and stores it in the specified path. If no training was done and the loaded model
        was a Sentence Transformer model already, then its model card is reused.

        Args:
            path (str): The path where the model card will be stored.
            model_name (str | None, optional): The name of the model. Defaults to None.
            train_datasets (list[str] | None, optional): Deprecated argument. Defaults to "deprecated".

        Returns:
            None
        """
        if model_name:
            model_path = Path(model_name)
            if not model_path.exists() and not self.model_card_data.model_id:
                self.model_card_data.model_id = model_name

        # If we loaded a Sentence Transformer model from the Hub, and no training was done, then
        # we don't generate a new model card, but reuse the old one instead.
        if self._model_card_text and self.model_card_data.trainer is None:
            model_card = self._model_card_text
        else:
            try:
                model_card = generate_model_card(self)
            except Exception:
                logger.error(
                    f"Error while generating model card:\n{traceback.format_exc()}"
                    "Consider opening an issue on https://github.com/UKPLab/sentence-transformers/issues with this traceback.\n"
                    "Skipping model card creation."
                )
                return

        with open(os.path.join(path, "README.md"), "w", encoding="utf8") as fOut:
            fOut.write(model_card)

    @save_to_hub_args_decorator
    def save_to_hub(
        self,
        repo_id: str,
        organization: str | None = None,
        token: str | None = None,
        private: Optional[bool] = None,
        safe_serialization: bool = True,
        commit_message: str = "Add new SentenceTransformer model.",
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
    ) -> str:
        """
        DEPRECATED, use `push_to_hub` instead.

        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

        Args:
            repo_id (str): Repository name for your model in the Hub, including the user or organization.
            token (str, optional): An authentication token (See https://huggingface.co/settings/token)
            private (bool, optional): Set to true, for hosting a private model
            safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way
            commit_message (str, optional): Message to commit while pushing.
            local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
            exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
            replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card
            train_datasets (list[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.

        Returns:
            str: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        logger.warning(
            "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers."
            " Please use `push_to_hub` instead for future model uploads."
        )

        if organization:
            if "/" not in repo_id:
                logger.warning(
                    f'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="{organization}/{repo_id}"` instead.'
                )
                repo_id = f"{organization}/{repo_id}"
            elif repo_id.split("/")[0] != organization:
                raise ValueError(
                    "Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
                )
            else:
                logger.warning(
                    f'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="{repo_id}"` instead.'
                )

        return self.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            safe_serialization=safe_serialization,
            commit_message=commit_message,
            local_model_path=local_model_path,
            exist_ok=exist_ok,
            replace_model_card=replace_model_card,
            train_datasets=train_datasets,
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: Optional[bool] = None,
        safe_serialization: bool = True,
        commit_message: str = "Add new SentenceTransformer model.",
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
    ) -> str:
        """
        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

        Args:
            repo_id (str): Repository name for your model in the Hub, including the user or organization.
            token (str, optional): An authentication token (See https://huggingface.co/settings/token)
            private (bool, optional): Set to true, for hosting a private model
            safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way
            commit_message (str, optional): Message to commit while pushing.
            local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
            exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
            replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card
            train_datasets (list[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.

        Returns:
            str: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=None,
            exist_ok=exist_ok,
        )
        repo_id = repo_url.repo_id  # Update the repo_id in case the old repo_id didn't contain a user or organization
        self.model_card_data.set_model_id(repo_id)
        if local_model_path:
            folder_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=local_model_path,
                commit_message=commit_message,
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                create_model_card = replace_model_card or not os.path.exists(
                    os.path.join(tmp_dir, "README.md")
                )
                self.save(
                    tmp_dir,
                    model_name=repo_url.repo_id,
                    create_model_card=create_model_card,
                    train_datasets=train_datasets,
                    safe_serialization=safe_serialization,
                )
                folder_url = api.upload_folder(
                    repo_id=repo_id, folder_path=tmp_dir, commit_message=commit_message
                )

        refs = api.list_repo_refs(repo_id=repo_id)
        for branch in refs.branches:
            if branch.name == "main":
                return f"https://huggingface.co/{repo_id}/commit/{branch.target_commit}"
        # This isn't expected to ever be reached.
        return folder_url

    @staticmethod
    def _text_length(text: list[int] | list[list[int]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None):
        """
        Evaluate the model based on an evaluator

        Args:
            evaluator (SentenceEvaluator): The evaluator used to evaluate the model.
            output_path (str, optional): The path where the evaluator can write the results. Defaults to None.

        Returns:
            The evaluation results.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

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
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (bool | str | None): The token to use for the model.
            cache_folder (str | None): The folder to cache the model.
            revision (str | None, optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (dict | None, optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (dict | None, optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            list[nn.Module]: A list containing the transformer model and the pooling model.
        """
        logger.warning(
            f"No sentence-transformers model found with name {model_name_or_path}. Creating a ColBERT model from base encoder."
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
            model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
        )
        self.model_card_data.set_base_model(model_name_or_path, revision=revision)
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
    ) -> dict[str, nn.Module]:
        """
        Loads a full SentenceTransformer model using the modules.json file.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (bool | str | None): The token to use for the model.
            cache_folder (str | None): The folder to cache the model.
            revision (str | None, optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (dict | None, optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (dict | None, optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            OrderedDict[str, nn.Module]: An ordered dictionary containing the modules of the model.
        """
        logger.warning(
            f"Creating a ColBERT model from a sentence-transformers model with name {model_name_or_path}."
        )
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_sentence_transformers_json_path is not None:
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

            if (
                "__version__" in self._model_config
                and "sentence_transformers" in self._model_config["__version__"]
                and self._model_config["__version__"]["sentence_transformers"]
                > __version__
            ):
                logger.warning(
                    "You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(
                        self._model_config["__version__"]["sentence_transformers"],
                        __version__,
                    )
                )

            # Set score functions & prompts if not already overridden by the __init__ calls
            if self.similarity_fn_name is None:
                self.similarity_fn_name = self._model_config.get(
                    "similarity_fn_name", None
                )
            if not self.prompts:
                self.prompts = self._model_config.get("prompts", {})
            if not self.default_prompt_name:
                self.default_prompt_name = self._model_config.get(
                    "default_prompt_name", None
                )
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

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path,
            "README.md",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if model_card_path is not None:
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        # Load the modules of sentence transformer
        modules_json_path = load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            module_class = import_from_string(module_config["type"])
            # For Transformer, don't load the full directory, rely on `transformers` instead
            # But, do load the config file first.
            if module_class == Transformer and module_config["path"] == "":
                kwargs = {}
                for config_name in [
                    "sentence_bert_config.json",
                    "sentence_roberta_config.json",
                    "sentence_distilbert_config.json",
                    "sentence_camembert_config.json",
                    "sentence_albert_config.json",
                    "sentence_xlm-roberta_config.json",
                    "sentence_xlnet_config.json",
                ]:
                    config_path = load_file_path(
                        model_name_or_path,
                        config_name,
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    if config_path is not None:
                        with open(config_path) as fIn:
                            kwargs = json.load(fIn)
                            # Don't allow configs to set trust_remote_code
                            if (
                                "model_args" in kwargs
                                and "trust_remote_code" in kwargs["model_args"]
                            ):
                                kwargs["model_args"].pop("trust_remote_code")
                            if (
                                "tokenizer_args" in kwargs
                                and "trust_remote_code" in kwargs["tokenizer_args"]
                            ):
                                kwargs["tokenizer_args"].pop("trust_remote_code")
                            if (
                                "config_args" in kwargs
                                and "trust_remote_code" in kwargs["config_args"]
                            ):
                                kwargs["config_args"].pop("trust_remote_code")
                        break

                hub_kwargs = {
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                    "local_files_only": local_files_only,
                }
                # 3rd priority: config file
                if "model_args" not in kwargs:
                    kwargs["model_args"] = {}
                if "tokenizer_args" not in kwargs:
                    kwargs["tokenizer_args"] = {}
                if "config_args" not in kwargs:
                    kwargs["config_args"] = {}

                # 2nd priority: hub_kwargs
                kwargs["model_args"].update(hub_kwargs)
                kwargs["tokenizer_args"].update(hub_kwargs)
                kwargs["config_args"].update(hub_kwargs)

                # 1st priority: kwargs passed to SentenceTransformer
                if model_kwargs:
                    kwargs["model_args"].update(model_kwargs)
                if tokenizer_kwargs:
                    kwargs["tokenizer_args"].update(tokenizer_kwargs)
                if config_kwargs:
                    kwargs["config_args"].update(config_kwargs)

                module = Transformer(
                    model_name_or_path, cache_dir=cache_folder, **kwargs
                )
            else:
                # Normalize does not require any files to be loaded
                if module_class == Normalize:
                    module_path = None
                else:
                    module_path = load_dir_path(
                        model_name_or_path,
                        module_config["path"],
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                module = module_class.load(module_path)
            modules[module_config["name"]] = module

        if revision is None:
            path_parts = Path(modules_json_path)
            if len(path_parts.parts) >= 2:
                revision_path_part = Path(modules_json_path).parts[-2]
                if len(revision_path_part) == 40:
                    revision = revision_path_part
        self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        # We only keep the Transformer/LinearProjection modules in case of sentence_transformers model to prune the potential PoolingLayer
        modules = [
            module
            for module in modules.values()
            if isinstance(module, Transformer) or isinstance(module, LinearProjection)
        ]
        return modules

    @staticmethod
    def load(input_path):
        return ColBERT(input_path)

    @property
    def device(self) -> torch.device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        In case there are no PyTorch parameters, fall back to CPU.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(
                module: nn.Module,
            ) -> list[tuple[str, torch.Tensor]]:
                tuples = [
                    (k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)
                ]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            try:
                first_tuple = next(gen)
                return first_tuple[1].device
            except StopIteration:
                return torch.device("cpu")

    @property
    def tokenizer(self):
        """
        Property to get the tokenizer that is used by this model
        """
        return self._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Property to set the tokenizer that should be used by this model
        """
        self._first_module().tokenizer = value

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximal input sequence length for the model. Longer inputs will be truncated.

        Returns:
            int: The maximal input sequence length.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-mpnet-base-v2")
                print(model.max_seq_length)
                # => 384
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value

    @property
    def _target_device(self) -> torch.device:
        logger.warning(
            "`SentenceTransformer._target_device` has been deprecated, please use `SentenceTransformer.device` instead.",
        )
        return self.device

    @_target_device.setter
    def _target_device(
        self, device: Optional[Union[int, str, torch.device]] = None
    ) -> None:
        self.to(device)

    @property
    def _no_split_modules(self) -> list[str]:
        try:
            return self._first_module()._no_split_modules
        except AttributeError:
            return []

    @property
    def _keys_to_ignore_on_save(self) -> list[str]:
        try:
            return self._first_module()._keys_to_ignore_on_save
        except AttributeError:
            return []
