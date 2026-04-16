from __future__ import annotations

import logging
import os

import torch
from safetensors import safe_open
from sentence_transformers.base.modules import Dense as DenseSentenceTransformer
from sentence_transformers.util import fullname, import_from_string
from torch import nn
from transformers.utils import cached_file

__all__ = ["Dense"]

logger = logging.getLogger(__name__)


class Dense(DenseSentenceTransformer):
    """Performs linear projection on the token embeddings to a lower dimension.

    Parameters
    ----------
    in_features
        Size of the embeddings in output of the transformer.
    out_features
        Size of the output embeddings after linear projection
    bias
        Add a bias vector
    activation_function
        Activation function applied after the linear layer.
    init_weight
        Initial value for the matrix of the linear layer
    init_bias
        Initial value for the bias of the linear layer.
    use_residual
        Whether to use residual for the linear layer.

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

    config_keys: list[str] = [
        "in_features",
        "out_features",
        "bias",
        "activation_function",
        "module_input_name",
        "module_output_name",
        "use_residual",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function=nn.Identity(),
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
        use_residual: bool = False,
        module_input_name: str = "token_embeddings",
        module_output_name: str | None = None,
    ) -> None:
        super(Dense, self).__init__(
            in_features,
            out_features,
            bias,
            activation_function,
            init_weight,
            init_bias,
            module_input_name=module_input_name,
            module_output_name=module_output_name or module_input_name,
        )
        self.use_residual = use_residual
        if use_residual and self.in_features != self.out_features:
            self.residual = nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs linear projection on the token embeddings."""
        token_embeddings = features[self.module_input_name]
        projected_embeddings = self.activation_function(self.linear(token_embeddings))
        if self.use_residual:
            residual_embeddings = token_embeddings
            if self.in_features != self.out_features:
                residual_embeddings = self.residual(token_embeddings)
            projected_embeddings = projected_embeddings + residual_embeddings
        features[self.module_output_name] = projected_embeddings
        return features

    @staticmethod
    def from_sentence_transformers(dense: DenseSentenceTransformer) -> "Dense":
        """Converts a SentenceTransformer Dense model to a Dense model.
        Our Dense model does not have the activation function.
        """
        config = dense.get_config_dict()
        if "activation_function" in config:
            act_fn = config["activation_function"]
            if isinstance(act_fn, str):
                config["activation_function"] = import_from_string(act_fn)()
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
        model_name_or_path
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        cache_folder
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        token
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only
            If `True`, will only try to load the tokenizer configuration from local files.
        use_auth_token
            Deprecated, use `token` instead.
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
                logging.warning("No safetensor model found, falling back to bin.")
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
                logging.warning("No safetensor model found, falling back to bin.")
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

    def get_config_dict(self):
        config = super().get_config_dict()
        config["activation_function"] = fullname(self.activation_function)
        return config

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "Dense":
        """Load a Dense layer."""
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        if "activation_function" in config:
            if trust_remote_code or config["activation_function"].startswith("torch."):
                config["activation_function"] = import_from_string(config["activation_function"])()
            else:
                logger.warning(
                    f"Activation function path '{config['activation_function']}' is not trusted, "
                    "falling back to the default activation function (Identity). "
                    "Please load the model with `trust_remote_code=True` to allow loading custom activation "
                    "functions via the configuration."
                )
                del config["activation_function"]
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
