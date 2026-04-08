"""Test suite for WARP index."""

import shutil
import uuid

import pytest

from pylate import indexes, models

pytest.importorskip("xtr_warp")


@pytest.fixture()
def warp_index(request):
    """Create a WARP index with test-friendly defaults and clean up afterwards.

    Pass extra kwargs via ``@pytest.mark.parametrize`` indirect or
    ``request.param``; otherwise uses aggressive search parameters to
    ensure full recall on tiny test indexes.
    """
    extra = getattr(request, "param", {}) or {}
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    defaults = dict(
        index_folder=folder,
        index_name=f"warp_{random_hash}",
        override=True,
        device="cpu",
        search=indexes.WARPSearchConfig(nprobe=32, centroid_score_threshold=0.0),
        indexing=indexes.WARPIndexingConfig(nbits=2, kmeans_niters=1),
    )
    defaults.update(extra)
    index = indexes.WARP(**defaults)
    yield index, folder
    shutil.rmtree(folder, ignore_errors=True)


def _make_model():
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )


def test_warp_add_and_search(warp_index):
    """Test basic add + search workflow."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["0", "1", "2", "3", "4"], documents_embeddings=embeddings
    )

    query_embedding = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 5, "Should return all 5 documents"
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"0", "1", "2", "3", "4"}


def test_warp_delete_single(warp_index):
    """Test deleting a single document."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["0", "1", "2", "3", "4"], documents_embeddings=embeddings
    )

    index.remove_documents(["1"])

    query_embedding = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 4
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"0", "2", "3", "4"}


def test_warp_delete_multiple_at_once(warp_index):
    """Test deleting multiple documents in one call."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document A about nutrition.",
        "Document B about health.",
        "Document C about wellness.",
        "Document D about fitness.",
        "Document E about diet.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C", "D", "E"], documents_embeddings=embeddings
    )

    index.remove_documents(["B", "D"])

    query_embedding = model.encode(["health and wellness"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 3
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"A", "C", "E"}


def test_warp_delete_sequential(warp_index):
    """Test multiple sequential deletions."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["0", "1", "2", "3", "4"], documents_embeddings=embeddings
    )

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


def test_warp_delete_and_add(warp_index):
    """Test that adding documents after deletion works correctly."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "First document about fruits.",
        "Second document about vegetables.",
        "Third document about grains.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["1", "2", "3"], documents_embeddings=embeddings)

    index.remove_documents(["2"])

    new_doc = ["Fourth document about legumes."]
    new_embedding = model.encode(new_doc, is_query=False)
    index.add_documents(documents_ids=["4"], documents_embeddings=new_embedding)

    query_embedding = model.encode(["food and nutrition"], is_query=True)
    matches = index(query_embedding, k=10)

    assert len(matches[0]) == 3
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"1", "3", "4"}


def test_warp_delete_nonexistent(warp_index):
    """Test that deleting a non-existent document doesn't cause errors."""
    index, _ = warp_index
    model = _make_model()

    documents = ["Document 1", "Document 2"]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(documents_ids=["1", "2"], documents_embeddings=embeddings)

    index.remove_documents(["999"])

    query_embedding = model.encode(["document"], is_query=True)
    matches = index(query_embedding, k=10)
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"1", "2"}


def test_warp_reload_after_delete():
    """Test that a fresh WARP instance correctly loads state after deletion."""
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    name = f"warp_{random_hash}"

    search_cfg = indexes.WARPSearchConfig(nprobe=32, centroid_score_threshold=0.0)
    indexing_cfg = indexes.WARPIndexingConfig(nbits=2, kmeans_niters=1)

    try:
        index = indexes.WARP(
            index_folder=folder,
            index_name=name,
            override=True,
            device="cpu",
            search=search_cfg,
            indexing=indexing_cfg,
        )
        model = _make_model()

        documents = [
            "Document X about machine learning.",
            "Document Y about artificial intelligence.",
            "Document Z about deep learning.",
        ]
        embeddings = model.encode(documents, is_query=False)
        index.add_documents(
            documents_ids=["X", "Y", "Z"], documents_embeddings=embeddings
        )

        index.remove_documents(["Y"])
        del index

        # Reload from disk
        index = indexes.WARP(
            index_folder=folder,
            index_name=name,
            override=False,
            device="cpu",
            search=search_cfg,
            indexing=indexing_cfg,
        )

        query_embedding = model.encode(["AI and ML"], is_query=True)
        matches = index(query_embedding, k=10)

        returned_ids = {m["id"] for m in matches[0]}
        assert returned_ids == {"X", "Z"}

        del index
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_warp_subset_search(warp_index):
    """Test searching with a subset filter."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples.",
        "Document about bananas.",
        "Document about cherries.",
        "Document about dates.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C", "D"], documents_embeddings=embeddings
    )

    query_embedding = model.encode(["fruit"], is_query=True)

    # Search only within subset
    matches = index(query_embedding, k=10, subset=["A", "C"])
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids <= {"A", "C"}, (
        f"Results should only contain subset documents, got {returned_ids}"
    )


def test_warp_update_documents(warp_index):
    """Test updating document embeddings in-place."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C"], documents_embeddings=embeddings
    )

    # Replace B's embedding with something about space
    new_embedding = model.encode(
        ["Document about rockets and space exploration."], is_query=False
    )
    index.update_documents(documents_ids=["B"], documents_embeddings=new_embedding)

    # B should now rank higher for a space query than a fruit query
    space_query = model.encode(["rockets and space"], is_query=True)
    matches = index(space_query, k=3)
    returned_ids = [m["id"] for m in matches[0]]
    assert "B" in returned_ids

    # All three documents should still be present
    fruit_query = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(fruit_query, k=10)
    assert {m["id"] for m in matches[0]} == {"A", "B", "C"}


def test_warp_compact(warp_index):
    """Test that compact works after deletions."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples.",
        "Document about bananas.",
        "Document about cherries.",
        "Document about dates.",
        "Document about elderberries.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C", "D", "E"], documents_embeddings=embeddings
    )

    index.remove_documents(["B", "D"])
    index.compact()

    query_embedding = model.encode(["fruit"], is_query=True)
    matches = index(query_embedding, k=10)
    returned_ids = {m["id"] for m in matches[0]}
    assert returned_ids == {"A", "C", "E"}


def test_warp_per_query_subset(warp_index):
    """Test searching with per-query subset filters."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples.",
        "Document about bananas.",
        "Document about cherries.",
        "Document about dates.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C", "D"], documents_embeddings=embeddings
    )

    queries = model.encode(["apples", "dates"], is_query=True)

    # Each query gets its own subset
    matches = index(queries, k=10, subset=[["A", "B"], ["C", "D"]])
    assert {m["id"] for m in matches[0]} <= {"A", "B"}
    assert {m["id"] for m in matches[1]} <= {"C", "D"}


@pytest.mark.parametrize(
    "warp_index",
    [{"dtype": __import__("torch").float32, "mmap": True}],
    indirect=True,
)
def test_warp_dtype_and_mmap(warp_index):
    """Test that dtype and mmap parameters are accepted and work."""
    index, _ = warp_index
    model = _make_model()

    documents = [
        "Document about apples.",
        "Document about bananas.",
        "Document about cherries.",
    ]
    embeddings = model.encode(documents, is_query=False)
    index.add_documents(
        documents_ids=["A", "B", "C"], documents_embeddings=embeddings
    )

    query_embedding = model.encode(["fruit"], is_query=True)
    matches = index(query_embedding, k=10)
    assert {m["id"] for m in matches[0]} == {"A", "B", "C"}
