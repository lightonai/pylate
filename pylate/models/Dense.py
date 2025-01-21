from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model
from sentence_transformers.models import Dense as DenseSentenceTransformer
from sentence_transformers.util import import_from_string
from torch import nn
from transformers.utils import cached_file

__all__ = ["Dense"]


class Dense(DenseSentenceTransformer):
    """Performs linear projection on the token embeddings to a lower dimension.

    Parameters
    ----------
    in_features
        Size of the embeddings in output of the tansformer.
    out_features
        Size of the output embeddings after linear projection
    bias
        Add a bias vector
    init_weight
        Initial value for the matrix of the linear layer
    init_bias
        Initial value for the bias of the linear layer.

    Examples
    --------
    >>> from pylate import models

    >>> model = models.Dense(
    ...     in_features=768,
    ...     out_features=128,
    ... )

    >>> features = {
    ...     "token_embeddings": torch.randn(2, 768),
    ... }

    >>> projected_features = model(features)

    >>> assert projected_features["token_embeddings"].shape == (2, 128)
    >>> assert isinstance(model, DenseSentenceTransformer)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function=nn.Identity(),
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
    ) -> None:
        super(Dense, self).__init__(
            in_features, out_features, bias, activation_function, init_weight, init_bias
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs linear projection on the token embeddings."""
        token_embeddings = features["token_embeddings"]
        projected_embeddings = self.linear(token_embeddings)
        features["token_embeddings"] = projected_embeddings
        return features

    @staticmethod
    def from_sentence_transformers(dense: DenseSentenceTransformer) -> "Dense":
        """Converts a SentenceTransformer Dense model to a Dense model.
        Our Dense model does not have the activation function.
        """
        config = dense.get_config_dict()
        config["activation_function"] = nn.Identity()
        model = Dense(**config)
        model.load_state_dict(dense.state_dict())
        return model

    @staticmethod
    def from_stanford_weights(
        model_name_or_path: str | os.PathLike,
        cache_folder: str | os.PathLike | None = None,
        revision: str | None = None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
        use_auth_token: str | bool | None = None,
    ) -> "Dense":
        """Load the weight of the Dense layer using weights from a stanford-nlp checkpoint.

        Parameters
        ----------
        model_name_or_path (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        cache_folder (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        """
        # Check if the model is locally available
        if not (os.path.exists(os.path.join(model_name_or_path))):
            # Else download the model/use the cached version. We first try to use the safetensors version and fall back to bin if not existing. All the recent stanford-nlp models are safetensors but we keep bin for compatibility.
            try:
                model_name_or_path = cached_file(
                    model_name_or_path,
                    filename="model.safetensors",
                    cache_dir=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                    token=token,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError:
                print("No safetensor model found, falling back to bin.")
                model_name_or_path = cached_file(
                    model_name_or_path,
                    filename="pytorch_model.bin",
                    cache_dir=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                    token=token,
                    use_auth_token=use_auth_token,
                )
        # If the model a local folder, load the safetensor
        # Again, we first try to load the safetensors version and fall back to bin if not existing.
        else:
            if os.path.exists(os.path.join(model_name_or_path, "model.safetensors")):
                model_name_or_path = os.path.join(
                    model_name_or_path, "model.safetensors"
                )
            else:
                print("No safetensor model found, falling back to bin.")
                model_name_or_path = os.path.join(
                    model_name_or_path, "pytorch_model.bin"
                )
        if model_name_or_path.endswith("safetensors"):
            with safe_open(model_name_or_path, framework="pt", device="cpu") as f:
                state_dict = {"linear.weight": f.get_tensor("linear.weight")}
        else:
            state_dict = {
                "linear.weight": torch.load(model_name_or_path, map_location="cpu")[
                    "linear.weight"
                ]
            }

        # Determine input and output dimensions
        in_features = state_dict["linear.weight"].shape[1]
        out_features = state_dict["linear.weight"].shape[0]

        # Create Dense layer instance
        model = Dense(in_features=in_features, out_features=out_features, bias=False)

        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def load(input_path) -> "Dense":
        """Load a Dense layer."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["activation_function"] = import_from_string(
            config["activation_function"]
        )()

        model = Dense(**config)

        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
            return model

        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
