from typing import Optional, List, Union
from abc import abstractmethod


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
    
    def get_doc_embeddings(self, doc_ids: List[str]) -> List[List[Union[int, float]]]:
        pass

 