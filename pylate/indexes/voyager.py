from __future__ import annotations

import itertools
import os

import numpy as np
import torch
from sqlitedict import SqliteDict
from voyager import Index, Space

from ..utils import iter_batch
from .base import Base


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reshape the embeddings, the Voyager index expects arrays with shape batch_size, n_tokens, embedding_size."""
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    if isinstance(embeddings, torch.Tensor):
        return reshape_embeddings(embeddings=embeddings.cpu().detach().numpy())

    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        return [embedding.cpu().detach().numpy() for embedding in embeddings]

    return embeddings


class Voyager(Base):
    """Voyager index. The Voyager index is a fast and efficient index for approximate nearest neighbor search.

    Parameters
    ----------
    name
        The name of the collection.
    override
        Whether to override the collection if it already exists.
    embedding_size
        The number of dimensions of the embeddings.
    M
        The number of subquantizers.
    ef_construction
        The number of candidates to evaluate during the construction of the index.
    ef_search
        The number of candidates to evaluate during the search.

    Examples
    --------
    >>> from pylate import indexes, models

    >>> index = indexes.Voyager(
    ...     index_folder="test_indexes",
    ...     index_name="colbert",
    ...     override=True,
    ...     embedding_size=128,
    ... )

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ... )

    >>> documents_embeddings = model.encode(
    ...     ["fruits are healthy.", "fruits are good for health.", "fruits are bad for health."],
    ...     is_query=False,
    ... )

    >>> index = index.add_documents(
    ...     documents_ids=["1", "2", "3"],
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["fruits are healthy.", "fruits are good for health and fun."],
    ...     is_query=True,
    ... )

    >>> matchs = index(queries_embeddings, k=30)

    >>> assert matchs["distances"].shape[0] == 2
    >>> assert isinstance(matchs, dict)
    >>> assert "documents_ids" in matchs
    >>> assert "distances" in matchs

    >>> queries_embeddings = model.encode(
    ...     "fruits are healthy.",
    ...     is_query=True,
    ... )

    >>> matchs = index(queries_embeddings, k=30)

    >>> assert matchs["distances"].shape[0] == 1
    >>> assert isinstance(matchs, dict)
    >>> assert "documents_ids" in matchs
    >>> assert "distances" in matchs
    >>> index = indexes.Voyager(
    ...     index_folder="test_indexes",
    ...     index_name="colbert",
    ...     override=False,
    ... )
    >>> matchs = index(queries_embeddings, k=30)
    >>> assert isinstance(matchs, dict)
    >>> assert "documents_ids" in matchs
    >>> assert "distances" in matchs
    >>> index = index.remove_documents(
    ...     documents_ids=["1"],
    ... )
    >>> matchs = index(queries_embeddings, k=30)
    >>> assert isinstance(matchs, dict)
    >>> assert "documents_ids" in matchs
    >>> assert "distances" in matchs
    >>> index = index.add_documents(
    ...     documents_ids=["1"],
    ...     documents_embeddings=documents_embeddings[0],
    ... )
    >>> matchs = index(queries_embeddings, k=30)
    >>> assert isinstance(matchs, dict)
    >>> assert "documents_ids" in matchs
    >>> assert "distances" in matchs



    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "colbert",
        override: bool = False,
        embedding_size: int = 128,
        M: int = 64,
        ef_construction: int = 200,
        ef_search: int = 200,
    ) -> None:
        self.ef_search = ef_search

        if not os.path.exists(path=index_folder):
            os.makedirs(name=index_folder)

        self.index_path = os.path.join(index_folder, f"{index_name}.voyager")
        self.documents_ids_to_embeddings_path = os.path.join(
            index_folder, f"{index_name}_document_ids_to_embeddings.sqlite"
        )
        self.embeddings_to_documents_ids_path = os.path.join(
            index_folder, f"{index_name}_embeddings_to_documents_ids.sqlite"
        )

        self.index = self._create_collection(
            index_path=self.index_path,
            embedding_size=embedding_size,
            M=M,
            ef_constructions=ef_construction,
            override=override,
        )

    def _load_documents_ids_to_embeddings(self) -> SqliteDict:
        """Load the SQLite database that maps document IDs to embeddings."""
        return SqliteDict(self.documents_ids_to_embeddings_path, outer_stack=False)

    def _load_embeddings_to_documents_ids(self) -> SqliteDict:
        """Load the SQLite database that maps embeddings to document IDs."""
        return SqliteDict(self.embeddings_to_documents_ids_path, outer_stack=False)

    def _create_collection(
        self,
        index_path: str,
        embedding_size: int,
        M: int,
        ef_constructions: int,
        override: bool,
    ) -> None:
        """Create a new Voyager collection.

        Parameters
        ----------
        index_path
            The path to the index.
        embedding_size
            The size of the embeddings.
        M
            The number of subquantizers.
        ef_constructions
            The number of candidates to evaluate during the construction of the index.
        override
            Whether to override the collection if it already exists.

        """
        if os.path.exists(path=index_path) and not override:
            with open(index_path, "rb") as f:
                return Index.load(f)

        if os.path.exists(path=index_path):
            os.remove(index_path)

        # Create the Voyager index
        index = Index(
            Space.Cosine,
            num_dimensions=embedding_size,
            M=M,
            ef_construction=ef_constructions,
        )

        index.save(index_path)

        if override and os.path.exists(path=self.documents_ids_to_embeddings_path):
            os.remove(path=self.documents_ids_to_embeddings_path)

        if override and os.path.exists(path=self.embeddings_to_documents_ids_path):
            os.remove(path=self.embeddings_to_documents_ids_path)

        # Create the SQLite databases
        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        documents_ids_to_embeddings.close()

        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        embeddings_to_documents_ids.close()

        return index

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        batch_size: int = 2000,
    ) -> None:
        """Add documents to the index."""
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        """Add documents to the index."""
        documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)

        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        for document_embeddings_batch, documents_ids_batch in zip(
            iter_batch(
                documents_embeddings,
                batch_size,
                desc=f"Adding documents to the index (bs={batch_size})",
            ),
            iter_batch(documents_ids, batch_size, tqdm_bar=False),
        ):
            embeddings_ids = self.index.add_items(
                list(itertools.chain(*document_embeddings_batch))
            )

            total = 0
            for doc_id, document_embeddings in zip(
                documents_ids_batch, document_embeddings_batch
            ):
                document_embeddings_ids = embeddings_ids[
                    total : total + len(document_embeddings)
                ]
                documents_ids_to_embeddings[doc_id] = document_embeddings_ids

                embeddings_to_documents_ids.update(
                    dict.fromkeys(document_embeddings_ids, doc_id)
                )
                total += len(document_embeddings)

        documents_ids_to_embeddings.commit()
        documents_ids_to_embeddings.close()

        embeddings_to_documents_ids.commit()
        embeddings_to_documents_ids.close()
        self.index.save(self.index_path)
        return self

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Remove documents from the index.

        Parameters
        ----------
        documents_ids
            The documents IDs to remove.

        """
        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()

        for document_id in documents_ids:
            embeddings_ids = documents_ids_to_embeddings[document_id]
            for embedding_id in embeddings_ids:
                del embeddings_to_documents_ids[embedding_id]
                self.index.mark_deleted(embedding_id)
            del documents_ids_to_embeddings[document_id]

        documents_ids_to_embeddings.commit()
        embeddings_to_documents_ids.commit()

        documents_ids_to_embeddings.close()
        embeddings_to_documents_ids.close()
        self.index.save(self.index_path)
        return self

    def __call__(
        self,
        queries_embeddings: np.ndarray | torch.Tensor,
        k: int = 10,
    ) -> dict:
        """Query the index for the nearest neighbors of the queries embeddings.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of nearest neighbors to return.

        """
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        k = min(k, len(embeddings_to_documents_ids))

        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
        n_queries = len(queries_embeddings)

        indices, distances = self.index.query(
            list(itertools.chain(*queries_embeddings)), k, query_ef=self.ef_search
        )

        if len(indices) == 0:
            raise ValueError("Index is empty, add documents before querying.")

        documents = [
            [
                [
                    embeddings_to_documents_ids[str(token_indice)]
                    for token_indice in tokens_indices
                ]
                for tokens_indices in document_indices
            ]
            for document_indices in indices.reshape(n_queries, -1, k)
        ]

        embeddings_to_documents_ids.close()

        return {
            "documents_ids": documents,
            "distances": distances.reshape(n_queries, -1, k),
        }

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Retrieve document embeddings for re-ranking from Voyager."""

        # Load mappings of document IDs to embedding IDs
        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()

        # Retrieve embedding IDs in the same structure as document IDs
        embedding_ids_structure = [
            [documents_ids_to_embeddings[doc_id] for doc_id in doc_group]
            for doc_group in document_ids
        ]

        documents_ids_to_embeddings.close()

        # Flatten the embedding IDs for a single API call
        flattened_embedding_ids = list(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(embedding_ids_structure)
            )
        )

        # Retrieve all embeddings in one API call
        all_embeddings = self.index.get_vectors(flattened_embedding_ids)

        # Reconstruct embeddings into the original structure
        reconstructed_embeddings = []
        embedding_idx = 0
        for group_embedding_ids in embedding_ids_structure:
            group_embeddings = []
            for doc_embedding_ids in group_embedding_ids:
                num_embeddings = len(doc_embedding_ids)
                group_embeddings.append(
                    all_embeddings[embedding_idx : embedding_idx + num_embeddings]
                )
                embedding_idx += num_embeddings
            reconstructed_embeddings.append(group_embeddings)

        return reconstructed_embeddings
