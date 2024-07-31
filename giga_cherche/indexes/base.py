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
    def create_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def add_documents(
        self, doc_ids: list[str], doc_embeddings: list[list[list[int | float]]]
    ) -> None:
        pass

    @abstractmethod
    def remove_documents(self, doc_ids: list[str]) -> None:
        pass

    @abstractmethod
    def query(self, queries_embeddings: list[list[int | float]], k: int = 5):
        pass

    @abstractmethod
    def get_docs_embeddings(
        self, doc_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        pass
