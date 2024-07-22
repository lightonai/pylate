import asyncio
import time

import weaviate
import weaviate.classes as wvc

from .base import Base

__all__ = ["Weaviate"]


# TODO: define Index metaclass
# max_doc_length is used to set a limit in the fetch embeddings method as the speed is dependant on the number of embeddings fetched
class Weaviate(Base):
    def __init__(
        self,
        host: str | None = "localhost",
        port: str | None = "8080",
        name: str | None = "colbert_collection",
        recreate: bool = False,
        max_doc_length: int | None = 180,
    ) -> None:
        self.host = host
        self.port = port
        self.name = name
        self.max_doc_length = max_doc_length
        fail_counter = 0
        attempt_number = 5
        retry_delay = 5.0
        while fail_counter < attempt_number:
            try:
                with weaviate.connect_to_local(
                    host=self.host, port=self.port
                ) as client:
                    print("Successful connection to the Weaviate container.")
                    if not client.collections.exists(self.name):
                        print(f"Collection {self.name} does not exist, creating it.")
                        self.create_collection(self.name)
                    elif recreate:
                        print(f"Collection {self.name} exists, recreating it.")
                        client.collections.delete(self.name)
                        self.create_collection(self.name)
                    else:
                        print(
                            f"Loaded collection with {client.collections.get(self.name).aggregate.over_all(total_count=True).total_count} vectors",
                        )

                    break
            except Exception as e:
                print(
                    f"Could not connect to the Weaviate container, retrying in {retry_delay} secs: {str(e)}"
                )
                fail_counter += 1
                time.sleep(retry_delay)

        if fail_counter >= attempt_number:
            raise ConnectionError("Could not connect to the Weaviate container")

    def create_collection(self, name: str) -> None:
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            client.collections.create(
                name=name,
                # TODO: let the user decide the type of index?
                # vector_index_config=wvc.config.Configure.VectorIndex.flat(
                #     distance_metric=wvc.config.VectorDistances.COSINE
                # ),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
                properties=[
                    wvc.config.Property(
                        name="doc_id", data_type=wvc.config.DataType.TEXT
                    ),
                ],
            )

    # TODO: embeddings could be a list of numpy array
    def add_documents(
        self, doc_ids: list[str], doc_embeddings: list[list[list[int | float]]]
    ) -> None:
        # assert we have the same number of doc_ids and doc_embeddings
        assert len(doc_ids) == len(doc_embeddings)
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            vector_index = client.collections.get(self.name)
            # data_objects = []
            # for doc_id, tokens_embeddings in zip(doc_ids, doc_embeddings):
            #     for token_embedding in tokens_embeddings:
            #         data_objects.append(
            #             wvc.data.DataObject(
            #                 properties={"doc_id": doc_id},
            #                 vector=token_embedding,
            #             )
            #         )
            # TODO: use dynamic batching insert
            data_objects = [
                wvc.data.DataObject(
                    properties={"doc_id": doc_id}, vector=token_embedding.tolist()
                )
                for doc_id, tokens_embeddings in zip(doc_ids, doc_embeddings)
                for token_embedding in tokens_embeddings
            ]
            vector_index.data.insert_many(data_objects)

    def remove_documents(self, doc_ids: list[str]) -> None:
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            vector_index = client.collections.get(self.name)
            vector_index.data.delete_many(
                where=wvc.query.Filter.by_property("doc_id").contains_any(doc_ids)
            )

    # TODO: add return type
    async def query_embedding(self, vector_index, query_embedding, k):
        return await vector_index.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            include_vector=True,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

    async def query_embeddings(self, vector_index, query_embeddings, k) -> list:
        tasks = [
            self.query_embedding(
                vector_index=vector_index, query_embedding=query_embedding, k=k
            )
            for query_embedding in query_embeddings
        ]
        return await asyncio.gather(*tasks)

    async def query_all_embeddings(
        self, queries_embeddings: list[list[int, float]], k: int = 5
    ) -> dict:
        async with weaviate.use_async_with_local() as client:
            vector_index = client.collections.get(self.name)
            tasks = [
                self.query_embeddings(
                    vector_index=vector_index, query_embeddings=query_embeddings, k=k
                )
                for query_embeddings in queries_embeddings
            ]
            res_queries = await asyncio.gather(*tasks)
            res = {}

            res["embeddings"] = [
                [[o.vector["default"] for o in obj.objects] for obj in res_query]
                for res_query in res_queries
            ]
            res["doc_ids"] = [
                [[o.properties["doc_id"] for o in obj.objects] for obj in res_query]
                for res_query in res_queries
            ]

            res["distances"] = [
                [[o.metadata.distance for o in obj.objects] for obj in res_query]
                for res_query in res_queries
            ]
            return res

    def query(self, queries_embeddings: list[list[int | float]], k: int = 5):
        return asyncio.run(self.query_all_embeddings(queries_embeddings, k))

    async def get_doc_embeddings(self, vector_index, doc_id: str):
        return await vector_index.query.fetch_objects(
            filters=wvc.query.Filter.by_property("doc_id").equal(doc_id),
            include_vector=True,
            limit=self.max_doc_length,
        )

    async def get_query_doc_embeddings(
        self, vector_index, query_doc_ids: list[str]
    ) -> list:
        tasks = [
            self.get_doc_embeddings(vector_index, doc_id) for doc_id in query_doc_ids
        ]
        return await asyncio.gather(*tasks)

    async def get_all_doc_embeddings(
        self, doc_ids: list[list[str]]
    ) -> list[list[list]]:
        # for query_doc_ids in doc_ids:
        #     for doc_id in query_doc_ids:
        #         print(doc_id)
        async with weaviate.use_async_with_local() as client:
            vector_index = client.collections.get(self.name)
            tasks = [
                self.get_query_doc_embeddings(vector_index, query_doc_ids)
                for query_doc_ids in doc_ids
            ]
            res_docs = await asyncio.gather(*tasks)
            return [
                [
                    [doc.vector["default"] for doc in document.objects]
                    for document in res_doc
                ]
                for res_doc in res_docs
            ]

    def get_docs_embeddings(
        self, doc_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        return asyncio.run(self.get_all_doc_embeddings(doc_ids))