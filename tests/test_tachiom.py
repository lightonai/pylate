"""Test suite for the TachiomIndex backend."""

import shutil
import uuid

import numpy as np
import pytest

from pylate import indexes, models

pytest.importorskip("tachiom")

# PQ codebook training requires at least 256 tokens in the corpus, hence the
# long documents.
DOCUMENTS = [
    "Apples are among the most widely cultivated tree fruits in the world. They are an "
    "excellent source of dietary fiber and vitamin C, and their regular consumption has "
    "been associated with improved heart health, better digestion and a reduced risk of "
    "type 2 diabetes. The skin of the apple contains most of its antioxidants.",
    "Bananas are a staple food in many tropical regions and one of the most popular "
    "fruits worldwide. They are particularly rich in potassium and vitamin B6, which "
    "support muscle function and energy metabolism. Their natural sugars and soft "
    "texture make them a convenient snack before or after physical exercise.",
    "Cherries are small stone fruits prized for their deep red color and sweet-tart "
    "flavor. They contain anthocyanins and other antioxidants that have been linked to "
    "reduced inflammation and muscle soreness. Tart cherry juice is also a natural "
    "source of melatonin, which may help regulate sleep cycles.",
    "Dates are the fruit of the date palm and have been cultivated in the Middle East "
    "for thousands of years. They are very high in natural sugars, making them an "
    "efficient source of quick energy, and they also provide fiber, potassium and "
    "magnesium. Dates are often used as a natural sweetener in baking.",
    "Elderberries are dark purple berries traditionally used in syrups and teas to "
    "support the immune system. They are rich in vitamin C and flavonoids, and several "
    "studies suggest elderberry extract may reduce the duration and severity of common "
    "cold and flu symptoms. Raw elderberries must be cooked before eating.",
]

DOCUMENT_IDS = ["0", "1", "2", "3", "4"]


@pytest.fixture()
def tachiom_index(request):
    """Create a TachiomIndex and clean up its folder afterwards.

    Pass extra kwargs via ``@pytest.mark.parametrize`` indirect;
    otherwise uses the corpus-adaptive defaults.
    """
    extra = getattr(request, "param", {}) or {}
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    defaults = dict(
        index_folder=folder,
        index_name=f"tachiom_{random_hash}",
        override=True,
    )
    defaults.update(extra)
    index = indexes.TachiomIndex(**defaults)
    yield index, folder
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope="module")
def model():
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )


def _build_index(index, model, documents=DOCUMENTS, documents_ids=DOCUMENT_IDS):
    """Encode documents with token IDs and build the index."""
    embeddings = model.encode(documents, is_query=False, output_value=None)
    index.add_documents(documents_ids=documents_ids, documents_embeddings=embeddings)
    return index


def test_tachiom_add_and_search(tachiom_index, model):
    """Test the basic add + search workflow with token IDs (TAC enabled)."""
    index, _ = tachiom_index
    _build_index(index, model)

    queries_embeddings = model.encode(
        ["fruits and nutrition", "berries and immune support"], is_query=True
    )
    matches = index(queries_embeddings, k=10)

    assert isinstance(matches, list)
    assert len(matches) == 2
    assert len(matches[0]) == 5, "Should return all 5 documents"
    assert matches[0][0].keys() == {"id", "score"}
    returned_ids = {match["id"] for match in matches[0]}
    assert returned_ids == set(DOCUMENT_IDS)

    # Scores must be sorted in decreasing order.
    for query_matches in matches:
        scores = [match["score"] for match in query_matches]
        assert scores == sorted(scores, reverse=True)


def test_tachiom_search_input_shapes(tachiom_index, model):
    """Test that __call__ accepts a single 2D array, a list of 2D arrays and a 3D array."""
    index, _ = tachiom_index
    _build_index(index, model)

    queries_embeddings = model.encode(
        ["fruits and nutrition", "berries and immune support"], is_query=True
    )

    # List of 2D arrays.
    matches = index(queries_embeddings, k=3)
    assert len(matches) == 2

    # Single 2D array is treated as one query.
    matches_single = index(queries_embeddings[0], k=3)
    assert len(matches_single) == 1
    assert [m["id"] for m in matches_single[0]] == [m["id"] for m in matches[0]]

    # List of (1, n_tokens, dim) arrays, as returned by encode(padding=True).
    padded_embeddings = model.encode(
        ["fruits and nutrition", "berries and immune support"],
        is_query=True,
        padding=True,
    )
    matches_padded = index(padded_embeddings, k=3)
    assert len(matches_padded) == 2

    # 3D array (n_queries, n_tokens, dim).
    queries_3d = np.vstack([np.asarray(q) for q in padded_embeddings])
    assert queries_3d.ndim == 3
    matches_3d = index(queries_3d, k=3)
    assert len(matches_3d) == 2
    for query_matches, query_matches_3d in zip(matches_padded, matches_3d):
        assert [m["id"] for m in query_matches] == [m["id"] for m in query_matches_3d]


def test_tachiom_explicit_token_ids(tachiom_index, model):
    """Test passing plain embeddings together with explicit documents_token_ids."""
    index, _ = tachiom_index

    encoded = model.encode(DOCUMENTS, is_query=False, output_value=None)
    embeddings = [
        emb[mask] for emb, mask in zip(encoded["token_embeddings"], encoded["masks"])
    ]
    token_ids = [
        ids[mask].astype(np.uint32)
        for ids, mask in zip(encoded["input_ids"], encoded["masks"])
    ]

    index.add_documents(
        documents_ids=DOCUMENT_IDS,
        documents_embeddings=embeddings,
        documents_token_ids=token_ids,
    )

    queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(queries_embeddings, k=10)
    assert {match["id"] for match in matches[0]} == set(DOCUMENT_IDS)


def test_tachiom_add_without_token_ids_warns(tachiom_index, model):
    """Test that omitting token IDs warns (TAC degrades to global k-means) but still works."""
    index, _ = tachiom_index
    embeddings = model.encode(DOCUMENTS, is_query=False)

    with pytest.warns(UserWarning, match="documents_token_ids not provided"):
        index.add_documents(documents_ids=DOCUMENT_IDS, documents_embeddings=embeddings)

    queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(queries_embeddings, k=10)
    assert {match["id"] for match in matches[0]} == set(DOCUMENT_IDS)


def test_tachiom_second_add_is_noop(tachiom_index, model):
    """Test that a second add_documents call warns and leaves the index unchanged."""
    index, _ = tachiom_index
    _build_index(index, model)

    queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
    matches_before = index(queries_embeddings, k=10)

    new_embeddings = model.encode(
        ["Document about figs and fiber."], is_query=False, output_value=None
    )
    with pytest.warns(UserWarning, match="already been built"):
        index.add_documents(documents_ids=["5"], documents_embeddings=new_embeddings)

    matches_after = index(queries_embeddings, k=10)
    assert {match["id"] for match in matches_after[0]} == set(DOCUMENT_IDS)
    assert [m["id"] for m in matches_after[0]] == [m["id"] for m in matches_before[0]]


def test_tachiom_remove_documents_is_noop(tachiom_index, model):
    """Test that remove_documents warns and leaves the index unchanged."""
    index, _ = tachiom_index
    _build_index(index, model)

    with pytest.warns(UserWarning, match="does not currently support document removal"):
        index.remove_documents(["1"])

    queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
    matches = index(queries_embeddings, k=10)
    assert {match["id"] for match in matches[0]} == set(DOCUMENT_IDS)


def test_tachiom_reload(model):
    """Test that a fresh TachiomIndex instance loads the index from disk."""
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    name = f"tachiom_{random_hash}"

    try:
        index = indexes.TachiomIndex(
            index_folder=folder, index_name=name, override=True
        )
        _build_index(index, model)

        queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
        matches_before = index(queries_embeddings, k=10)
        del index

        index = indexes.TachiomIndex(
            index_folder=folder, index_name=name, override=False
        )
        assert index.is_indexed

        matches_after = index(queries_embeddings, k=10)
        assert len(matches_after[0]) == 5
        assert [m["id"] for m in matches_after[0]] == [
            m["id"] for m in matches_before[0]
        ]

        del index
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_tachiom_get_documents_embeddings(tachiom_index, model):
    """Test that reconstructed embeddings match the originals in shape and direction."""
    index, _ = tachiom_index

    encoded = model.encode(DOCUMENTS, is_query=False, output_value=None)
    original_embeddings = [
        np.asarray(emb[mask], dtype=np.float32)
        for emb, mask in zip(encoded["token_embeddings"], encoded["masks"])
    ]
    index.add_documents(documents_ids=DOCUMENT_IDS, documents_embeddings=encoded)

    reconstructed = index.get_documents_embeddings([["0", "1"], ["2"]])

    assert len(reconstructed) == 2
    assert len(reconstructed[0]) == 2
    assert len(reconstructed[1]) == 1

    for original, approx in zip(
        [original_embeddings[0], original_embeddings[1], original_embeddings[2]],
        [reconstructed[0][0], reconstructed[0][1], reconstructed[1][0]],
    ):
        assert approx.shape == original.shape
        assert approx.dtype == np.float32
        # PQ reconstruction is lossy but should preserve token directions.
        cosines = np.sum(original * approx, axis=1) / (
            np.linalg.norm(original, axis=1) * np.linalg.norm(approx, axis=1)
        )
        assert float(cosines.mean()) > 0.8

    with pytest.raises((KeyError, ValueError)):
        index.get_documents_embeddings([["999"]])


def test_tachiom_empty_index_raises(tachiom_index, model):
    """Test that querying or reading an empty index raises a ValueError."""
    index, _ = tachiom_index

    queries_embeddings = model.encode(["fruits and nutrition"], is_query=True)
    with pytest.raises(ValueError, match="index is empty"):
        index(queries_embeddings, k=10)

    with pytest.raises(ValueError, match="index is empty"):
        index.get_documents_embeddings([["0"]])


def test_tachiom_add_documents_too_few_tokens_raises(tachiom_index, model):
    """Test that a corpus with too few tokens for PQ training raises ValueError."""
    index, _ = tachiom_index

    # Far fewer than the 256 tokens PQ codebook training requires.
    documents = [
        "Document about apples and their nutritional benefits.",
        "Document about bananas and their vitamin content.",
        "Document about cherries and antioxidants.",
        "Document about dates and natural sugars.",
        "Document about elderberries and immune support.",
    ]
    embeddings = model.encode(documents, is_query=False, output_value=None)

    with pytest.raises(ValueError, match="256"):
        index.add_documents(documents_ids=DOCUMENT_IDS, documents_embeddings=embeddings)

    # The failed build must not leave the index in a partially-built state.
    assert index.is_indexed is False
