import json
import os
from typing import Dict

import torch
from torch import Tensor, nn


# This is a linear projection layer, very similar to the Dense layer without a non-linearity and could be merged to it
class LinearProjection(nn.Module):
    """
    Performs linear projection on the token embeddings to a lower dimension.

    Args:
        in_features: Size of the word embeddings
        out_features: Size of the output embeddings after linear projection
        init_weight: Initial value for the matrix of the linear layer
        init_bias: Initial value for the bias of the linear layer


    Args:
        in_features: Size of the input dimension
        out_features: Output size
        bias: Add a bias vector
        activation_function: Pytorch activation function applied on
            output
        init_weight: Initial value for the matrix of the linear layer
        init_bias: Initial value for the bias of the linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_weight: Tensor = None,
        init_bias: Tensor = None,
    ):
        super(LinearProjection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear_projection = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]

        # Linear projection
        projected_embeddings = self.linear_projection(token_embeddings)
        # TODO: maybe we want to define features["projected_embeddings"] instead of overwriting token_embeddings?
        features["token_embeddings"] = projected_embeddings
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def __repr__(self):
        return "LinearProjection({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = LinearProjection(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
