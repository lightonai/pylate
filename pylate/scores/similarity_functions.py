from __future__ import annotations

from enum import Enum
from typing import Callable

from numpy import ndarray
from torch import Tensor

from ..scores import colbert_scores, colbert_scores_pairwise


class SimilarityFunction(Enum):
    """
    Enum class for supported score functions. The following functions are supported:
    - ``SimilarityFunction.MAXSIM`` (``"MaxSim"``): Max similarity
    """

    MAXSIM = "MaxSim"

    @staticmethod
    def to_similarity_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function name or enum value to the corresponding similarity function.

        Parameters
        ----------
        similarity_function
            The name or enum value of the similarity function.
        """
        similarity_function = SimilarityFunction(similarity_function)
        if similarity_function == SimilarityFunction.MAXSIM:
            return colbert_scores

        raise ValueError(
            f"The provided function {similarity_function} is not supported. Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function into a pairwise similarity function.

        The pairwise similarity function returns the diagonal vector from the similarity matrix, i.e. it only
        computes the similarity(a[i], b[i]) for each i in the range of the input tensors, rather than
        computing the similarity between all pairs of a and b.

        Parameters
        ----------
        similarity_function
            The name or enum value of the similarity function.
        """
        similarity_function = SimilarityFunction(similarity_function)
        if similarity_function == SimilarityFunction.MAXSIM:
            return colbert_scores_pairwise

    @staticmethod
    def possible_values() -> list[str]:
        """
        Returns a list of possible values for the SimilarityFunction enum.

        Examples
        --------

        >>> possible_values = SimilarityFunction.possible_values()
        >>> possible_values
        ['MaxSim']
        """
        return [m.value for m in SimilarityFunction]
