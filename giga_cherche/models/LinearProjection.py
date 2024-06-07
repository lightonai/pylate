import json
import os
from typing import Dict

import torch
from torch import Tensor, nn


class LinearProjection(nn.Module):
    """
    Performs linear projection on the token embeddings to a lower dimension.

    Args:
        word_embedding_dimension: Dimensions for the word embeddings
        output_dimension: Dimensions for the output embeddings after linear projection
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        output_dimension: int,
    ) -> None:
        super(LinearProjection, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "output_dimension",
        ]

        self.word_embedding_dimension = word_embedding_dimension
        self.output_dimension = output_dimension

        self.linear_projection = nn.Linear(word_embedding_dimension, output_dimension)

    def __repr__(self):
        return "LinearProjection({})".format(self.get_config_dict())

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]

        # Linear projection
        projected_embeddings = self.linear_projection(token_embeddings)
        # TODO: maybe we want to define features["projected_embeddings"] instead of overwriting token_embeddings?
        features["token_embeddings"] = projected_embeddings
        return features

    def get_sentence_embedding_dimension(self):
        return self.output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    #TODO: better saving/loading?
    def save(self, output_path, safe_serialization: bool = True):
        torch.save(self.linear_projection.state_dict(), output_path + "/linear_layer.bin")
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        module = LinearProjection(**config)
        module.linear_projection.load_state_dict(torch.load(input_path + "/linear_layer.bin"))
        return module