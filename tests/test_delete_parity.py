"""Parity tests asserting that ``FastPlaid`` and ``WARP`` produce the same
user-facing behavior across delete + re-add sequences.

The internal numeric passage IDs of the two engines diverge under deletion
(FastPlaid renumbers, WARP tombstones), but the user-facing string IDs the
caller passed to ``add_documents`` must remain stable and consistent across
both backends. These tests pin that contract so a future change to either
backend cannot silently break swap-in compatibility.
"""

import shutil
import uuid

import pytest

from pylate import indexes, models

pytest.importorskip("xtr_warp")


def _make_model():
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        model_kwargs={"attn_implementation": "eager"},
    )


def _make_index(backend: str, folder: str, name: str):
    if backend == "plaid":
        return indexes.PLAID(
            index_folder=folder,
            index_name=name,
            override=True,
            use_fast=True,
            nbits=2,
            kmeans_niters=1,
        )
    if backend == "warp":
        return indexes.WARP(
            index_folder=folder,
            index_name=name,
            override=True,
            device="cpu",
            n_ivf_probe=32,
            centroid_score_threshold=0.0,
            nbits=2,
            kmeans_niters=1,
        )
    raise ValueError(backend)


def _ids(matches):
    """Flatten a single-query ``__call__`` result into a set of doc IDs."""
    return {m["id"] for m in matches[0]}


@pytest.mark.parametrize("backend", ["plaid", "warp"])
def test_delete_keeps_higher_ids_addressable(backend):
    """Deleting a document with a lower passage-id must not invalidate
    higher-id documents on either backend.

    Scenario:
      1. Add A, B, C, D, E.
      2. Delete B (the second-lowest internal id).
      3. The remaining A, C, D, E must all be findable by their original
         user-facing ids — even though FastPlaid will renumber internally
         and WARP will tombstone.
      4. Add F. F must be findable; B must not reappear.
      5. Delete A (the lowest remaining). C, D, E, F must still be findable.
    """
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    name = f"{backend}_{random_hash}"

    try:
        index = _make_index(backend, folder, name)
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
            documents_ids=["A", "B", "C", "D", "E"],
            documents_embeddings=embeddings,
        )

        query = model.encode(["fruit nutrition"], is_query=True)

        # Step 1: all five present.
        assert _ids(index(query, k=10)) == {"A", "B", "C", "D", "E"}

        # Step 2: delete a lower-id document (B).
        index.remove_documents(["B"])
        # Step 3: higher-id documents stay addressable, B is gone.
        assert _ids(index(query, k=10)) == {"A", "C", "D", "E"}

        # Step 4: add a new document. Its user-facing id must work
        # regardless of how the backend assigns the internal id.
        f_embedding = model.encode(
            ["Document about figs and dietary fiber."], is_query=False
        )
        index.add_documents(documents_ids=["F"], documents_embeddings=f_embedding)
        assert _ids(index(query, k=10)) == {"A", "C", "D", "E", "F"}

        # Step 5: delete the lowest remaining id and check the rest.
        index.remove_documents(["A"])
        assert _ids(index(query, k=10)) == {"C", "D", "E", "F"}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize("backend", ["plaid", "warp"])
def test_delete_then_readd_same_id(backend):
    """A user-facing id that has been deleted must be re-usable: re-adding
    it should make it findable again on both backends, regardless of the
    internal renumbering scheme.
    """
    random_hash = uuid.uuid4().hex
    folder = f"test_indexes_{random_hash}"
    name = f"{backend}_{random_hash}"

    try:
        index = _make_index(backend, folder, name)
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

        index.remove_documents(["B"])

        # Re-add an id that was just deleted, with new content.
        new_embedding = model.encode(["Document about blueberries."], is_query=False)
        index.add_documents(documents_ids=["B"], documents_embeddings=new_embedding)

        query = model.encode(["fruit"], is_query=True)
        assert _ids(index(query, k=10)) == {"A", "B", "C"}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
