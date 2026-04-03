"""Test suite for WARP index."""

import shutil
import uuid

from pylate import indexes, models


def _make_index(**kwargs):
    """Create a WARP index with test-friendly defaults.

    Uses aggressive search parameters to ensure full recall on tiny test indexes.
    """
    random_hash = uuid.uuid4().hex
    defaults = dict(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"warp_{random_hash}",
        override=True,
        nbits=2,
        kmeans_niters=1,
        device="cpu",
        nprobe=32,
        centroid_score_threshold=0.0,
    )
    defaults.update(kwargs)
    return indexes.WARP(**defaults), defaults["index_folder"]


def _make_model():
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )


def test_warp_add_and_search():
    """Test basic add + search workflow."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["0", "1", "2", "3", "4"],
                        documents_embeddings=embeddings)

    query_embedding = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 5, "Should return all 5 documents"
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"0", "1", "2", "3", "4"}

    shutil.rmtree(folder)


def test_warp_delete_single():
    """Test deleting a single document."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["0", "1", "2", "3", "4"],
                        documents_embeddings=embeddings)

    index.remove_documents(["1"])

    query_embedding = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 4
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"0", "2", "3", "4"}

    shutil.rmtree(folder)


def test_warp_delete_multiple_at_once():
    """Test deleting multiple documents in one call."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "Document A about nutrition.",
        "Document B about health.",
        "Document C about wellness.",
        "Document D about fitness.",
        "Document E about diet.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["A", "B", "C", "D", "E"],
                        documents_embeddings=embeddings)

    index.remove_documents(["B", "D"])

    query_embedding = model.encode(["health and wellness"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 3
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"A", "C", "E"}

    shutil.rmtree(folder)


def test_warp_delete_sequential():
    """Test multiple sequential deletions."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["0", "1", "2", "3", "4"],
                        documents_embeddings=embeddings)

    query_embedding = model.encode(["fruits and nutrition"], is_query=True)

    index.remove_documents(["1"])
    matches = index(query_embedding, k=10)
    assert {m["id"] for m in matches[0]} == {"0", "2", "3", "4"}

    index.remove_documents(["3"])
    matches = index(query_embedding, k=10)
    assert {m["id"] for m in matches[0]} == {"0", "2", "4"}

    index.remove_documents(["0"])
    matches = index(query_embedding, k=10)
    assert {m["id"] for m in matches[0]} == {"2", "4"}

    shutil.rmtree(folder)


def test_warp_delete_and_add():
    """Test that adding documents after deletion works correctly."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "First document about fruits.",
        "Second document about vegetables.",
        "Third document about grains.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["1", "2", "3"],
                        documents_embeddings=embeddings)

    index.remove_documents(["2"])

    new_doc = ["Fourth document about legumes."]
    new_embedding = model.encode(new_doc, is_query=False)
    index.add_documents(documents_ids=["4"], documents_embeddings=new_embedding)

    query_embedding = model.encode(["food and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 3
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"1", "3", "4"}

    shutil.rmtree(folder)


def test_warp_delete_nonexistent():
    """Test that deleting a non-existent document doesn't cause errors."""
    index, folder = _make_index()
    model = _make_model()

    documents = ["Document 1", "Document 2"]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["1", "2"],
                        documents_embeddings=embeddings)

    index.remove_documents(["999"])

    query_embedding = model.encode(["document"], is_query=True)
    matches = index(query_embedding, k=10)
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"1", "2"}

    shutil.rmtree(folder)


def test_warp_reload_after_delete():
    """Test that a fresh WARP instance correctly loads state after deletion."""
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    name = f"warp_{random_hash}"

    index = indexes.WARP(
        index_folder=folder, index_name=name, override=True,
        nbits=2, kmeans_niters=1, device="cpu",
        nprobe=32, centroid_score_threshold=0.0,
    )
    model = _make_model()

    documents = [
        "Document X about machine learning.",
        "Document Y about artificial intelligence.",
        "Document Z about deep learning.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["X", "Y", "Z"],
                        documents_embeddings=embeddings)

    index.remove_documents(["Y"])
    del index

    # Reload from disk
    index = indexes.WARP(
        index_folder=folder, index_name=name, override=False,
        nbits=2, kmeans_niters=1, device="cpu",
        nprobe=32, centroid_score_threshold=0.0,
    )

    query_embedding = model.encode(["AI and ML"], is_query=True)
    matches = index(query_embedding, k=10)

    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"X", "Z"}

    del index
    shutil.rmtree(folder)


def test_warp_subset_search():
    """Test searching with a subset filter."""
    index, folder = _make_index()
    model = _make_model()

    documents = [
        "Document about apples.",
        "Document about bananas.",
        "Document about cherries.",
        "Document about dates.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["A", "B", "C", "D"],
                        documents_embeddings=embeddings)

    query_embedding = model.encode(["fruit"], is_query=True)

    # Search only within subset
    matches = index(query_embedding, k=10, subset=["A", "C"])
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids <= {"A", "C"}, (
        f"Results should only contain subset documents, got {returned_ids}"
    )

    shutil.rmtree(folder)


if __name__ == "__main__":
    test_warp_add_and_search()
    test_warp_delete_single()
    test_warp_delete_multiple_at_once()
    test_warp_delete_sequential()
    test_warp_delete_and_add()
    test_warp_delete_nonexistent()
    test_warp_reload_after_delete()
    test_warp_subset_search()
