"""Test suite for FastPlaid index delete and re-indexing logic."""

import shutil
import uuid

from pylate import indexes, models


def test_fast_plaid_delete_reindexing():
    """Test that FastPlaid correctly re-indexes IDs after deletion.

    This test ensures that after deleting documents, the remaining documents
    maintain correct ID mappings. Fast-plaid re-assigns IDs starting from 0
    after deletion, so we need to verify the mapping layer correctly handles this.
    """
    random_hash = uuid.uuid4().hex

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,  # Lower bits for faster test
        kmeans_niters=1,  # Fewer iterations for faster test
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    # Create 5 documents with distinct content
    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]

    documents_embeddings = model.encode(documents, is_query=False)

    # Add all documents with IDs 0-4
    document_ids = ["0", "1", "2", "3", "4"]
    index.add_documents(
        documents_ids=document_ids, documents_embeddings=documents_embeddings
    )

    # Query to find all documents
    query = "fruits and nutrition"
    query_embedding = model.encode([query], is_query=True)

    # Verify all 5 documents are in the index
    matches = index(query_embedding, k=10)
    assert len(matches[0]) == 5, "Should have 5 documents initially"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"0", "1", "2", "3", "4"}, (
        "All document IDs should be present"
    )

    # Delete document "1" (middle position)
    index.remove_documents(["1"])

    # Query again - should only have 4 documents now
    matches = index(query_embedding, k=10)
    assert len(matches[0]) == 4, "Should have 4 documents after deleting one"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"0", "2", "3", "4"}, "Document '1' should be removed"
    assert "1" not in returned_ids, "Deleted document should not appear"

    # Delete document "3" (another middle position)
    index.remove_documents(["3"])

    matches = index(query_embedding, k=10)
    assert len(matches[0]) == 3, "Should have 3 documents after deleting two"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"0", "2", "4"}, "Only documents '0', '2', '4' should remain"

    # Delete document "0" (first position)
    index.remove_documents(["0"])

    matches = index(query_embedding, k=10)
    assert len(matches[0]) == 2, "Should have 2 documents after deleting three"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"2", "4"}, "Only documents '2', '4' should remain"

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


def test_fast_plaid_delete_multiple_at_once():
    """Test deleting multiple documents at once."""
    random_hash = uuid.uuid4().hex

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,
        kmeans_niters=1,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    documents = [
        "Document A about nutrition.",
        "Document B about health.",
        "Document C about wellness.",
        "Document D about fitness.",
        "Document E about diet.",
    ]

    documents_embeddings = model.encode(documents, is_query=False)
    document_ids = ["A", "B", "C", "D", "E"]

    index.add_documents(
        documents_ids=document_ids, documents_embeddings=documents_embeddings
    )

    query_embedding = model.encode(["health and wellness"], is_query=True)

    # Delete multiple documents at once: B and D
    index.remove_documents(["B", "D"])

    matches = index(query_embedding, k=10)
    assert len(matches[0]) == 3, "Should have 3 documents after deleting two"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"A", "C", "E"}, "Documents A, C, E should remain"

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


def test_fast_plaid_delete_and_add_new():
    """Test that adding documents after deletion works correctly."""
    random_hash = uuid.uuid4().hex

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,
        kmeans_niters=1,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    # Add initial documents
    documents = [
        "First document about fruits.",
        "Second document about vegetables.",
        "Third document about grains.",
    ]

    documents_embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["1", "2", "3"], documents_embeddings=documents_embeddings
    )

    # Delete document "2"
    index.remove_documents(["2"])

    # Add new document "4"
    new_doc = ["Fourth document about legumes."]
    new_embedding = model.encode(new_doc, is_query=False)
    index.add_documents(documents_ids=["4"], documents_embeddings=new_embedding)

    # Query and verify all expected documents are present
    query_embedding = model.encode(["food and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 3, "Should have 3 documents (1, 3, 4)"
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"1", "3", "4"}, "Documents 1, 3, 4 should be present"
    assert "2" not in returned_ids, "Deleted document 2 should not be present"

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


def test_fast_plaid_delete_edge_cases():
    """Test edge cases in deletion: first, last, and consecutive deletions."""
    random_hash = uuid.uuid4().hex

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,
        kmeans_niters=1,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    documents = [
        f"Document {i} with unique content about topic {i}." for i in range(10)
    ]

    documents_embeddings = model.encode(documents, is_query=False)
    document_ids = [str(i) for i in range(10)]

    index.add_documents(
        documents_ids=document_ids, documents_embeddings=documents_embeddings
    )

    query_embedding = model.encode(["document content"], is_query=True)

    # Test deleting last document
    index.remove_documents(["9"])
    matches = index(query_embedding, k=15)
    returned_ids = {match["id"] for match in matches[0]}
    assert "9" not in returned_ids
    assert len(matches[0]) == 9

    # Test deleting consecutive documents (2, 3, 4)
    index.remove_documents(["2", "3", "4"])
    matches = index(query_embedding, k=15)
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"0", "1", "5", "6", "7", "8"}
    assert len(matches[0]) == 6

    # Test deleting first document
    index.remove_documents(["0"])
    matches = index(query_embedding, k=15)
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"1", "5", "6", "7", "8"}
    assert len(matches[0]) == 5

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


def test_fast_plaid_delete_nonexistent():
    """Test that deleting a non-existent document doesn't cause errors."""
    random_hash = uuid.uuid4().hex

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,
        kmeans_niters=1,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    documents = ["Document 1", "Document 2"]
    documents_embeddings = model.encode(documents, is_query=False)

    index.add_documents(
        documents_ids=["1", "2"], documents_embeddings=documents_embeddings
    )

    # Try to delete non-existent document - should not raise error
    index.remove_documents(["999"])

    # Verify original documents are still there
    query_embedding = model.encode(["document"], is_query=True)
    matches = index(query_embedding, k=10)
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"1", "2"}

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


def test_fast_plaid_reload_after_delete():
    """Test that index can be reloaded after deletion and still works correctly."""
    random_hash = uuid.uuid4().hex

    # Create and populate index
    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=True,
        use_fast=True,
        nbits=2,
        kmeans_niters=1,
    )

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )

    documents = [
        "Document X about machine learning.",
        "Document Y about artificial intelligence.",
        "Document Z about deep learning.",
    ]

    documents_embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["X", "Y", "Z"], documents_embeddings=documents_embeddings
    )

    # Delete document Y
    index.remove_documents(["Y"])

    # Close and reload index
    del index

    index = indexes.PLAID(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"fast_plaid_{random_hash}",
        override=False,  # Load existing index
        use_fast=True,
    )

    # Query and verify correct documents remain
    query_embedding = model.encode(["AI and ML"], is_query=True)
    matches = index(query_embedding, k=10)

    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == {"X", "Z"}, "After reload, only X and Z should remain"
    assert "Y" not in returned_ids, (
        "Deleted document Y should not be present after reload"
    )

    # Cleanup
    del index
    shutil.rmtree(f"test_indexes_{random_hash}")


if __name__ == "__main__":
    # Run tests
    test_fast_plaid_delete_reindexing()

    test_fast_plaid_delete_multiple_at_once()

    test_fast_plaid_delete_and_add_new()

    test_fast_plaid_delete_edge_cases()

    test_fast_plaid_delete_nonexistent()

    test_fast_plaid_reload_after_delete()
