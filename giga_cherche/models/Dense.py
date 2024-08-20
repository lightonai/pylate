import json
import os

import torch
from safetensors.torch import load_model as load_safetensors_model
from sentence_transformers.models import Dense as DenseSentenceTransformer
from sentence_transformers.util import import_from_string
from torch import nn

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
    >>> from giga_cherche import models

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
    def from_sentence_transformers(dense_st: DenseSentenceTransformer):
        config = dense_st.get_config_dict()
        config["activation_function"] = nn.Identity()
        model = Dense(**config)
        model.load_state_dict(dense_st.state_dict())
        return model

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["activation_function"] = import_from_string(
            config["activation_function"]
        )()
        model = Dense(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"),
                    map_location=torch.device("cpu"),
                )
            )
        return model
