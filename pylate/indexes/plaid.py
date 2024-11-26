import itertools
import os

import numpy as np
import torch
from sqlitedict import SqliteDict

from .base import Base
from .stanford_nlp import Indexer, Searcher
from .stanford_nlp.infra import ColBERTConfig


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


class PLAID(Base):
    """PLAID index. The PLAID index is the most scalable type of index for multi-vector search and leverage PQ-IVF as well as custom kernel for decompression.

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

    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "colbert",
        override: bool = False,
        nbits: int = 2,
        nranks: int = 1,
        kmeans_niters: int = 4,
        bsize: int = 128,
        index_bsize: int = 1,
        doc_maxlen: int = 512,
    ) -> None:
        self.config = ColBERTConfig(
            nbits=nbits,
            nranks=nranks,
            root=f"{index_folder}",
            avoid_fork_if_possible=True,
            overwrite=override,
            kmeans_niters=kmeans_niters,
            bsize=bsize,
            index_bsize=index_bsize,
            doc_maxlen=doc_maxlen,
        )
        self.index_name = index_name
        self.indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=self.config)
        self.documents_ids_to_plaid_ids_path = os.path.join(
            index_folder, f"{index_name}_documents_ids_to_plaid_ids.sqlite"
        )
        self.plaid_ids_to_documents_ids_path = os.path.join(
            index_folder, f"{index_name}_plaid_ids_to_documents_ids.sqlite"
        )

    def _load_documents_ids_to_plaid_ids(self) -> SqliteDict:
        """Load the SQLite database that maps document IDs to PLAID IDs."""
        return SqliteDict(self.documents_ids_to_plaid_ids_path, outer_stack=False)

    def _load_plaid_ids_to_documents_ids(self) -> SqliteDict:
        """Load the SQLite database that maps PLAID IDs to document IDs."""
        return SqliteDict(self.plaid_ids_to_documents_ids_path, outer_stack=False)

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
        documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()
        self.indexer.index(
            name=f"{self.index_name}",
            collection=documents_embeddings,
            overwrite=True,
        )

        # Get total number of existing documents
        n_docs = len(documents_ids_to_plaid_ids)
        for i, document_id in enumerate(documents_ids):
            plaid_id = n_docs + i
            documents_ids_to_plaid_ids[document_id] = plaid_id
            plaid_ids_to_documents_ids[plaid_id] = document_id

        documents_ids_to_plaid_ids.commit()
        documents_ids_to_plaid_ids.close()

        plaid_ids_to_documents_ids.commit()
        plaid_ids_to_documents_ids.close()

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
        searcher = Searcher(index=self.index_name, config=self.config)
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()
        documents = []
        distances = []
        for query_embeddings in queries_embeddings:
            result = searcher.search(query_embeddings, k=k)
            documents.append([plaid_ids_to_documents_ids[r] for r in result[0]])
            distances.append(result[2])
        results = [
            [
                {"id": doc_id, "score": score}
                for doc_id, score in zip(query_documents, query_distances)
            ]
            for query_documents, query_distances in zip(documents, distances)
        ]
        return results
        return {
            "documents_ids": documents,
            "distances": distances,
        }

        # exit()
        # plaid_ids_to_documents_ids.close()
        # embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        # k = min(k, len(embeddings_to_documents_ids))

        # queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
        # n_queries = len(queries_embeddings)

        # indices, distances = self.index.query(
        #     list(itertools.chain(*queries_embeddings)), k, query_ef=self.ef_search
        # )

        # if len(indices) == 0:
        #     raise ValueError("Index is empty, add documents before querying.")

        # documents = [
        #     [
        #         [
        #             embeddings_to_documents_ids[str(token_indice)]
        #             for token_indice in tokens_indices
        #         ]
        #         for tokens_indices in document_indices
        #     ]
        #     for document_indices in indices.reshape(n_queries, -1, k)
        # ]

        # embeddings_to_documents_ids.close()

        return {
            "documents_ids": documents,
            "distances": distances,
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
