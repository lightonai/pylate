from __future__ import annotations

from abc import ABC, abstractmethod


class Base(ABC):
    """Base class for all indexes. Indexes are used to store and retrieve embeddings."""

    @abstractmethod
    def __init__(
        self,
        name: str | None = "colbert_collection",
        recreate: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[list[list[int | float]]],
        batch_size: int,
    ) -> None:
        pass

    @abstractmethod
    def remove_documents(self, documents_ids: list[str]) -> None:
        pass

    @abstractmethod
    def __call__(self, queries_embeddings: list[list[int | float]], k: int = 5):
        pass

    @abstractmethod
    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        pass
