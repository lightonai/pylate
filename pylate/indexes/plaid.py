import os

import numpy as np
import torch
from sqlitedict import SqliteDict

from .base import Base
from .stanford_nlp import Indexer, IndexUpdater, Searcher
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
        embedding_size: int = 128,
        nbits: int = 2,
        nranks: int = 1,
        kmeans_niters: int = 4,
        index_bsize: int = 1,
    ) -> None:
        self.config = ColBERTConfig(
            nbits=nbits,
            nranks=nranks,
            root=f"{index_folder}",
            index_root=f"{index_folder}",
            overwrite=override,
            kmeans_niters=kmeans_niters,
            index_bsize=index_bsize,
            dim=embedding_size,
        )
        self.index_name = index_name
        self.index_folder = index_folder
        self.indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=self.config)
        self.documents_ids_to_plaid_ids_path = os.path.join(
            index_folder, f"{index_name}_documents_ids_to_plaid_ids.sqlite"
        )
        self.plaid_ids_to_documents_ids_path = os.path.join(
            index_folder, f"{index_name}_plaid_ids_to_documents_ids.sqlite"
        )
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        if override:
            if os.path.exists(self.documents_ids_to_plaid_ids_path):
                os.remove(self.documents_ids_to_plaid_ids_path)
            if os.path.exists(self.plaid_ids_to_documents_ids_path):
                os.remove(self.plaid_ids_to_documents_ids_path)
            self.searcher = None
        else:
            documents_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
            # Check if the collection has already been populated and if so, load the searcher
            if len(documents_ids_to_plaid_ids) == 0:
                self.searcher = None
            else:
                self.searcher = Searcher(
                    index=self.index_name,
                    config=self.config,
                    index_root=f"{index_folder}/",
                )
            documents_ids_to_plaid_ids.close()

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
        if self.searcher is None:
            self.indexer.index(
                name=f"{self.index_name}",
                collection=documents_embeddings,
                overwrite=True,
            )
            self.searcher = Searcher(
                index=self.index_name,
                config=self.config,
                index_root=f"{self.index_folder}/",
            )
            plaid_ids = list(range(len(documents_embeddings)))
        else:
            index_updater = IndexUpdater(
                self.config, self.searcher, checkpoint="colbert-ir/colbertv2.0"
            )
            plaid_ids = index_updater.add(documents_embeddings)
            index_updater.persist_to_disk()

        # Get total number of existing documents
        for plaid_id, document_id in zip(plaid_ids, documents_ids):
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
        document_ids_to_plaid_ids = self._load_documents_ids_to_plaid_ids()
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()
        plaid_ids = [
            document_ids_to_plaid_ids[document_id] for document_id in documents_ids
        ]
        index_updater = IndexUpdater(
            self.config, self.searcher, checkpoint="colbert-ir/colbertv2.0"
        )
        index_updater.remove(plaid_ids)
        index_updater.persist_to_disk()
        for document_id in documents_ids:
            plaid_id = document_ids_to_plaid_ids[document_id]
            del document_ids_to_plaid_ids[document_id]
            del plaid_ids_to_documents_ids[plaid_id]

        document_ids_to_plaid_ids.commit()
        document_ids_to_plaid_ids.close()

        plaid_ids_to_documents_ids.commit()
        plaid_ids_to_documents_ids.close()

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
        plaid_ids_to_documents_ids = self._load_plaid_ids_to_documents_ids()
        documents = []
        distances = []
        # self.searcher = Searcher(index=self.index_name, config=self.config)
        for query_embeddings in queries_embeddings:
            result = self.searcher.search(query_embeddings, k=k)
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
        raise NotImplementedError
