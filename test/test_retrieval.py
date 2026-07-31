import rag_tasks

import pytest
from types import SimpleNamespace

from unittest.mock import patch
import vectorstore_chroma
import reranker_hf


def test_retrieve_happy_path():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), patch(
        "vectorstore_chroma.search", return_value=["chunk1", "chunk2"]
    ):
        result = rag_tasks.retrieve("what is homer's favorite food")

    assert isinstance(result, list)
    assert result == ["chunk1", "chunk2"]


def test_retrieve_whitespace_question_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        rag_tasks.retrieve(" ")


def test_retrieve_empty_chroma_results():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), patch(
        "vectorstore_chroma.search", return_value=[]
    ):
        result = rag_tasks.retrieve("obscure question with no matches")

    assert result == []


def test_retrieve_passes_k_to_chroma_search():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), patch(
        "vectorstore_chroma.search", return_value=[]
    ) as mock_search:
        rag_tasks.retrieve("some question", k=5)
    mock_search.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], k=5)
