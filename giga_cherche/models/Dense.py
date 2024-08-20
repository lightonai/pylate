import json
import os
from typing import Any

import torch
from torch import nn

__all__ = ["Dense"]
from sentence_transformers.models import Dense as DenseSentenceTransformer


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

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
    ) -> None:
        super(Dense, self).__init__(
            in_features, out_features, bias, init_weight, init_bias
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear_projection = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def __repr__(self) -> str:
        return f"Dense({self.get_config_dict()})"

    def __call__(self, features: dict[str, torch.Tensor]):
        return self.forward(features)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs linear projection on the token embeddings."""
        token_embeddings = features["token_embeddings"]
        projected_embeddings = self.linear_projection(token_embeddings)
        features["token_embeddings"] = projected_embeddings
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Returns the dimension of the sentence embeddings."""
        return self.out_features

    def get_config_dict(self) -> dict[str, Any]:
        """Returns the configuration of the model."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    def save(self, output_path: str) -> None:
        with open(file=os.path.join(output_path, "config.json"), mode="w") as fOut:
            json.dump(obj=self.get_config_dict(), fp=fOut)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path) -> "Dense":
        """Load the model from a directory."""
        with open(file=os.path.join(input_path, "config.json")) as file:
            config = json.load(file)

        model = Dense(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
