# Voyager

Voyager index. The Voyager index is a fast and efficient index for approximate nearest neighbor search.



## Parameters

- **index_folder** (*'str'*) – defaults to `indexes`

- **index_name** (*'str'*) – defaults to `colbert`

- **override** (*'bool'*) – defaults to `False`

    Whether to override the collection if it already exists.

- **embedding_size** (*'int'*) – defaults to `128`

    The number of dimensions of the embeddings.

- **M** (*'int'*) – defaults to `64`

    The number of subquantizers.

- **ef_construction** (*'int'*) – defaults to `200`

    The number of candidates to evaluate during the construction of the index.

- **ef_search** (*'int'*) – defaults to `200`

    The number of candidates to evaluate during the search.



## Examples

from pylate import indexes, models

index = indexes.Voyager(
    index_folder="test_indexes",
    index_name="colbert",
    override=True,
    embedding_size=128,
)

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
)

documents_embeddings = model.encode(
    ["fruits are healthy.", "fruits are good for health.", "fruits are bad for health."],
    is_query=False,
)

index = index.add_documents(
    documents_ids=["1", "2", "3"],
    documents_embeddings=documents_embeddings,
)

queries_embeddings = model.encode(
     ["fruits are healthy.", "fruits are good for health and fun."],
     is_query=True,
)

matches = index(queries_embeddings, k=30)

## Methods

???- note "__call__"

    Query the index for the nearest neighbors of the queries embeddings.

    **Parameters**

    - **queries_embeddings**     (*'np.ndarray | torch.Tensor'*)
    - **k**     (*'int'*)     – defaults to `10`

???- note "add_documents"

    Add documents to the index.

    **Parameters**

    - **documents_ids**     (*'str | list[str]'*)
    - **documents_embeddings**     (*'list[np.ndarray | torch.Tensor]'*)
    - **batch_size**     (*'int'*)     – defaults to `2000`

???- note "get_documents_embeddings"

    Retrieve document embeddings for re-ranking from Voyager.

    **Parameters**

    - **document_ids**     (*'list[list[str]]'*)

???- note "remove_documents"

    Remove documents from the index.

    **Parameters**

    - **documents_ids**     (*'list[str]'*)
