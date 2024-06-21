from abc import abstractmethod
from typing import List, Optional, Union


class BaseIndex:
    @abstractmethod
    def __init__(
        self,
        name: Optional[str] = "colbert_collection",
        recreate: Optional[bool] = False,
    ) -> None:
        pass

    @abstractmethod
    def create_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def add_documents(
        self, doc_ids: List[str], doc_embeddings: List[List[List[Union[int, float]]]]
    ) -> None:
        pass

    @abstractmethod
    def remove_documents(self, doc_ids: List[str]) -> None:
        pass

    # TODO: add return type
    @abstractmethod
    def query(self, queries_embeddings: List[List[Union[int, float]]], k: int = 5):
        pass

    @abstractmethod
    def get_docs_embeddings(
        self, doc_ids: List[List[str]]
    ) -> List[List[List[Union[int, float]]]]:
        pass
