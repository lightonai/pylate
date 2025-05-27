import torch
from scipy.cluster import hierarchy


def kmeans_pooling(
    documents_embeddings: list[torch.Tensor],
    pool_factor: int = 1,
    protected_tokens: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooled_embeddings = []

    if pool_factor > 1:
        for document_embeddings in documents_embeddings:
            document_length = len(document_embeddings)
            document_embeddings = document_embeddings.to(device=device)
            # Shuffle the embeddings within the passage
            document_embeddings = document_embeddings[torch.randperm(document_length)]
            document_pooled_embeddings = []
            norms = document_embeddings.norm(dim=1, keepdim=True)
            similarities = (
                document_embeddings @ document_embeddings.t() / (norms @ norms.t())
            )
            similarities.fill_diagonal_(-1)  # Exclude self-similarity
            unassigned_tokens = torch.ones(
                document_length, dtype=torch.bool, device=device
            )
            if protected_tokens > 0:
                # Mark the first `protected_tokens` as already assigned
                unassigned_tokens[:protected_tokens] = False
                document_pooled_embeddings.extend(
                    document_embeddings[:protected_tokens]
                )
            # Identify the 8 most unique tokens by their minimum similarity to others
            # if UNIQUE_KEEP > 0:
            #     min_similarities, _ = similarities.min(dim=1)
            #     unique_indices = torch.topk(
            #         min_similarities, UNIQUE_KEEP, largest=False
            #     ).indices
            #     unassigned_tokens[unique_indices] = (
            #         False  # Mark unique tokens as already assigned
            #     )
            #     for idx in unique_indices:
            #         document_pooled_embeddings.append(document_embeddings[idx])

            while unassigned_tokens.any():
                current_token = unassigned_tokens.nonzero(as_tuple=True)[0][0]
                top_similar_indices = similarities[current_token].argsort(
                    descending=True
                )[:pool_factor]
                pooled_group = torch.cat(
                    (current_token.unsqueeze(0), top_similar_indices)
                )
                unassigned_tokens[pooled_group] = False
                pooled_embedding = document_embeddings[pooled_group].mean(dim=0)
                document_pooled_embeddings.append(pooled_embedding)
            pooled_embeddings.append(torch.stack(tensors=document_pooled_embeddings))
        return pooled_embeddings
    else:
        return document_embeddings


def sequential_pooling(
    documents_embeddings: list[torch.Tensor],
    pool_factor: int = 1,
    protected_tokens: int = 1,
) -> list[torch.Tensor]:
    if pool_factor > 1:
        pooled_embeddings = []
        for document_embeddings in documents_embeddings:
            document_pooled_embeddings = []
            if protected_tokens > 0:
                # Protect the first `protected_tokens` embeddings
                document_pooled_embeddings.extend(
                    document_embeddings[:protected_tokens]
                )
                document_embeddings = document_embeddings[protected_tokens:]
            # Pool the rest of the embeddings
            document_pooled_embeddings.extend(
                document_embeddings[i : i + pool_factor].mean(dim=0)
                for i in range(0, len(document_embeddings), pool_factor)
            )
            pooled_embeddings.append(torch.stack(tensors=document_pooled_embeddings))
        return pooled_embeddings
    else:
        return documents_embeddings


def hierarchical_pooling(
    documents_embeddings: list[torch.Tensor],
    pool_factor: int = 1,
    protected_tokens: int = 1,
) -> list[torch.Tensor]:
    """
    Pools the embeddings hierarchically by clustering and averaging them.

    Parameters
    ----------
    document_embeddings_list
        A list of embeddings for each document.
    pool_factor
        Factor to determine the number of clusters. Defaults to 1.
    protected_tokens
        Number of tokens to protect from pooling at the start of each document. Defaults to 1.

    Returns
    -------
        A list of pooled embeddings for each document.
    """
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    pooled_embeddings = []

    for document_embeddings in documents_embeddings:
        document_embeddings = document_embeddings.to(device=device)

        # Separate protected tokens from the rest
        protected_embeddings = document_embeddings[:protected_tokens]
        embeddings_to_pool = document_embeddings[protected_tokens:]

        # Compute cosine similarity and convert to distance matrix
        cosine_similarities = torch.mm(
            input=embeddings_to_pool, mat2=embeddings_to_pool.t()
        )
        distance_matrix = 1 - cosine_similarities.cpu().numpy()

        # Perform hierarchical clustering using Ward's method
        clusters = hierarchy.linkage(distance_matrix, method="ward")
        num_embeddings = len(embeddings_to_pool)

        # Determine the number of clusters based on pool_factor
        num_clusters = max(num_embeddings // pool_factor, 1)
        cluster_labels = hierarchy.fcluster(
            clusters, t=num_clusters, criterion="maxclust"
        )

        # Pool embeddings within each cluster
        pooled_document_embeddings = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_indices = torch.where(
                condition=torch.tensor(data=cluster_labels == cluster_id, device=device)
            )[0]
            if cluster_indices.numel() > 0:
                cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
                pooled_document_embeddings.append(cluster_embedding)

        # Re-append protected embeddings
        pooled_document_embeddings.extend(protected_embeddings)
        pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))

    return pooled_embeddings


def late_chunking(
    documents_embeddings: list[torch.Tensor],
    chunk_size: int = 512,
) -> list[torch.Tensor]:
    """
    Pools the embeddings hierarchically by chunking and averaging them.

    Parameters
    ----------
    document_embeddings_list
        A list of embeddings for each document.
    chunk_size
        The size of chunks to be pooled
    Returns
    -------
        A list of pooled embeddings for each document.
    """
    pooled_embeddings = []
    for document_embeddings in documents_embeddings:
        document_pooled_embeddings = []
        for i in range(0, len(document_embeddings), chunk_size):
            chunk_embeddings = document_embeddings[i : i + chunk_size]
            pooled_chunk_embedding = chunk_embeddings.mean(dim=0)
            document_pooled_embeddings.append(pooled_chunk_embedding)

        pooled_embeddings.append(torch.stack(document_pooled_embeddings))

    return pooled_embeddings
