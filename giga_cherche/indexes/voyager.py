import itertools
import os
import shutil

from sqlitedict import SqliteDict
from voyager import Index, Space

from .base import Base


class Voyager(Base):
    def __init__(
        self,
        name: str = "colbert_collection",
        override_collection: bool = False,
        num_dimensions: int = 128,
        M: int = 64,
        ef_construction: int = 200,
        ef_search: int = 200,
    ):
        self.ef_search = ef_search
        self.folder_path = f"indexes/{name}"
        if not os.path.exists(self.folder_path):
            self.create_collection(self.folder_path, num_dimensions, M, ef_construction)
        elif override_collection:
            self.create_collection(self.folder_path, num_dimensions, M, ef_construction)
        else:
            self.index = Index.load(f"{self.folder_path}/index.voy")

    def create_collection(
        self, name: str, num_dimensions: int, M: int, ef_constructions: int
    ) -> None:
        self.index = Index(
            Space.Cosine,
            num_dimensions=num_dimensions,
            M=M,
            ef_construction=ef_constructions,
        )

        if os.path.exists(name):
            shutil.rmtree(name)
        os.makedirs(name)
        self.index.save(f"{name}/index.voy")
        doc_id_to_embeddings_ids = SqliteDict(
            f"{name}/doc_id_to_embeddings_ids.sqlite", outer_stack=False
        )
        doc_id_to_embeddings_ids.close()
        embeddings_id_to_doc_id = SqliteDict(
            f"{name}/embeddings_id_to_doc_id.sqlite", outer_stack=False
        )
        embeddings_id_to_doc_id.close()

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[list[list[int | float]]],
    ) -> None:
        doc_id_to_embeddings_ids = SqliteDict(
            f"{self.folder_path}/doc_id_to_embeddings_ids.sqlite", outer_stack=False
        )
        embeddings_id_to_doc_id = SqliteDict(
            f"{self.folder_path}/embeddings_id_to_doc_id.sqlite", outer_stack=False
        )
        # Add all the embeddings to the index at once
        embeddings_ids = self.index.add_items(
            list(itertools.chain(*documents_embeddings))
        )
        total = 0
        for doc_id, document_embeddings in zip(documents_ids, documents_embeddings):
            document_embeddings_ids = embeddings_ids[
                total : total + len(document_embeddings)
            ]
            doc_id_to_embeddings_ids[doc_id] = document_embeddings_ids

            embeddings_id_to_doc_id.update(
                dict.fromkeys(document_embeddings_ids, doc_id)
            )
            total += len(document_embeddings)
        doc_id_to_embeddings_ids.commit()
        embeddings_id_to_doc_id.commit()
        doc_id_to_embeddings_ids.close()
        embeddings_id_to_doc_id.close()

    def remove_documents(self, documents_ids: list[str]) -> None:
        doc_id_to_embeddings_ids = SqliteDict(
            f"{self.folder_path}/doc_id_to_embeddings_ids.sqlite", outer_stack=False
        )
        embeddings_id_to_doc_id = SqliteDict(
            f"{self.folder_path}/embeddings_id_to_doc_id.sqlite", outer_stack=False
        )
        for doc_id in documents_ids:
            embeddings_ids = doc_id_to_embeddings_ids[doc_id]
            for embedding_id in embeddings_ids:
                del embeddings_id_to_doc_id[embedding_id]
                self.index.mark_deleted(embedding_id)
            del doc_id_to_embeddings_ids[doc_id]
        doc_id_to_embeddings_ids.commit()
        embeddings_id_to_doc_id.commit()
        doc_id_to_embeddings_ids.close()
        embeddings_id_to_doc_id.close()

    def query(self, queries_embeddings: list[list[int | float]], k: int = 5):
        # Query the index for every embeddings at once
        indices, distances = self.index.query(
            list(itertools.chain(*queries_embeddings)), k, query_ef=self.ef_search
        )
        embeddings_id_to_doc_id = SqliteDict(
            f"{self.folder_path}/embeddings_id_to_doc_id.sqlite", outer_stack=False
        )

        indices = indices.reshape(len(queries_embeddings), -1, k)
        distances = distances.reshape(len(queries_embeddings), -1, k)

        res = {}
        res["doc_ids"] = [
            [
                [
                    embeddings_id_to_doc_id[str(token_indice)]
                    for token_indice in tokens_indices
                ]
                for tokens_indices in document_indices
            ]
            for document_indices in indices
        ]
        res["distances"] = distances
        embeddings_id_to_doc_id.close()
        return res

    def get_docs_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        doc_id_to_embeddings_ids = SqliteDict(
            f"{self.folder_path}/doc_id_to_embeddings_ids.sqlite", outer_stack=False
        )

        # Store embedding IDs in the original shape
        embeddings_ids = [
            [doc_id_to_embeddings_ids[doc_id] for doc_id in doc_ids]
            for doc_ids in documents_ids
        ]

        # Flatten the embedding IDs for a single API call
        all_embedding_ids = list(
            itertools.chain.from_iterable(itertools.chain.from_iterable(embeddings_ids))
        )

        # Make a single call to get all embeddings
        all_embeddings = self.index.get_vectors(all_embedding_ids)

        # Reshape using the original structure
        documents_embeddings = []
        embedding_index = 0
        for query_documents_embeddings_ids in embeddings_ids:
            query_documents_embeddings = []
            for query_document_embeddings_ids in query_documents_embeddings_ids:
                num_embeddings = len(query_document_embeddings_ids)
                document_embeddings = all_embeddings[
                    embedding_index : embedding_index + num_embeddings
                ]
                query_documents_embeddings.append(document_embeddings)
                embedding_index += num_embeddings
            documents_embeddings.append(query_documents_embeddings)

        doc_id_to_embeddings_ids.close()

        return documents_embeddings
