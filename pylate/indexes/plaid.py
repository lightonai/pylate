from __future__ import annotations

import logging
import os

import numpy as np
import torch
from sqlitedict import SqliteDict

from .base import Base
from .stanford_nlp import Indexer, IndexUpdater, Searcher
from .stanford_nlp.infra import ColBERTConfig

logger = logging.getLogger(__name__)


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reshape the embeddings, the PLAID index expects arrays with shape batch_size, n_tokens, embedding_size."""
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
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the collection if it already exists.
    embedding_size
        The number of dimensions of the embeddings.
    nbits
        The number of bits to use for the quantization.
    kmeans_niters
        The number of iterations to use for the k-means clustering.
    ndocs
        The number of candidate documents
    centroid_score_threshold
        The threshold scores for centroid pruning.
    ncells
        The number of cells to consider for search.
    search_batch_size
        The batch size to use when searching.

    Examples:
    --------
    from pylate import indexes, models

    index = indexes.PLAID(
        index_folder="indexes",
        index_name="plaid_colbert",
        override=True,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    )

    documents_embeddings = model.encode(
    [
        "The nutritional profile of fruits is exceptional, containing essential vitamins, minerals, and antioxidants that support immune function. Regular consumption of diverse fruits has been linked to reduced risk of chronic diseases including heart disease, type 2 diabetes, and certain cancers. The fiber content in fruits also promotes digestive health and helps maintain healthy cholesterol levels. Nutritionists recommend consuming at least 2-3 servings of fresh fruits daily as part of a balanced diet.",
        "Recent nutritional studies have challenged traditional views on fruit consumption. While fruits contain natural sugars, their impact on blood glucose levels varies significantly based on ripeness, variety, and processing methods. The glycemic index of fruits ranges widely, with berries typically scoring lower than tropical varieties. Individuals with metabolic conditions should consider these factors when incorporating fruits into their dietary plans.",
        "Paris is the capital of France and one of the most populous cities in Europe. It is known for its art, fashion, and culture. The city is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cuisine, with many world-renowned restaurants and cafes. The city has a rich history and has been a center of art and culture for centuries.",
    ],
        is_query=False,
    )

    index = index.add_documents(
        documents_ids=["1", "2", "3"], documents_embeddings=documents_embeddings
    )

    queries_embeddings = model.encode(
        ["not dangerous", "dangerous"],
        is_query=True,
    )

    index(queries_embeddings, k=30)

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
        ndocs: int = 8192,
        centroid_score_threshold: float = 0.35,
        ncells: int = 8,
        search_batch_size: int = 2**18,
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
            ndocs=ndocs,
            centroid_score_threshold=centroid_score_threshold,
            ncells=ncells,
            search_batch_size=search_batch_size,
        )
        self.index_name = index_name
        self.index_folder = index_folder
        self.indexer = Indexer(config=self.config)
        self.documents_ids_to_plaid_ids_path = os.path.join(
            index_folder, index_name, "documents_ids_to_plaid_ids.sqlite"
        )
        self.plaid_ids_to_documents_ids_path = os.path.join(
            index_folder, index_name, "plaid_ids_to_documents_ids.sqlite"
        )
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        if not os.path.exists(f"{index_folder}/{index_name}"):
            os.makedirs(f"{index_folder}/{index_name}")
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
            logger.warning(
                "You are adding documents to an existing index. This is an experimental feature and may result in suboptimal results. Please consider reindexing the entire collection (use override=True) or make sure to add as many documents during the first addition to provide the most accurate kmeans centroids."
            )
            index_updater = IndexUpdater(self.config, self.searcher)
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
        index_updater = IndexUpdater(self.config, self.searcher)
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
        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
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

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        raise NotImplementedError
