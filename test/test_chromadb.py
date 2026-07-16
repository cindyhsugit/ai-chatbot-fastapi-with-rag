from unittest.mock import patch, MagicMock
import pytest
import vectorstore_chroma


@patch("vectorstore_chroma.collection")
def test_search_happy_path(mock_collection):
    # happy path: collection has results, search correctly extracts
    # and returns the documents list
    mock_collection.query.return_value = {
        "documents": [["Homer Simpson's favorite food is broccoli casserole."]]
    }

    result = vectorstore_chroma.search(query_embedding=[0.1, 0.2, 0.3], k=5)

    assert result == ["Homer Simpson's favorite food is broccoli casserole."]
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]], n_results=5
    )


@patch("vectorstore_chroma.collection")
def test_search_error_path_empty_collection_raises(mock_collection):
    # error path: an empty collection returns documents=[[]] or
    # documents=[] depending on version — this documents the current
    # (crashing) behavior rather than assuming it's handled
    mock_collection.query.return_value = {"documents": []}

    with pytest.raises(IndexError):
        vectorstore_chroma.search(query_embedding=[0.1, 0.2, 0.3], k=5)


@patch("vectorstore_chroma.collection")
def test_search_edge_case_k_larger_than_available_documents(mock_collection):
    # edge case: k=20 requested, but the collection only has 2 documents
    # stored — Chroma itself handles this by just returning what it has,
    # so this confirms our wrapper doesn't choke on a short result either
    mock_collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]]
    }

    result = vectorstore_chroma.search(query_embedding=[0.1, 0.2, 0.3], k=20)

    assert result == ["chunk one", "chunk two"]