import rag_tasks
import numpy as np
import pytest
from unittest.mock import patch
import vectorstore_chroma
import reranker_hf

def test_retrieve_happy_path():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("vectorstore_chroma.search", return_value=["chunk1", "chunk2"]), \
         patch("reranker_hf.rerank", return_value=[("chunk1", 0.9), ("chunk2", 0.7)]) as mock_rerank:

        result = rag_tasks.retrieve("what is homer's favorite food")

    assert isinstance(result, list)
    assert result == [("chunk1", 0.9), ("chunk2", 0.7)]
    mock_rerank.assert_called_once_with("what is homer's favorite food", ["chunk1", "chunk2"], top_k=3)


def test_retrieve_whitespace_question_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        rag_tasks.retrieve(" ")


def test_retrieve_empty_chroma_results():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("vectorstore_chroma.search", return_value=[]), \
         patch("reranker_hf.rerank", return_value=[]) as mock_rerank:

        result = rag_tasks.retrieve("obscure question with no matches")

    assert result == []
    mock_rerank.assert_called_once_with("obscure question with no matches", [], top_k=3)


def test_retrieve_passes_k_to_chroma_search():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("vectorstore_chroma.search", return_value=[]) as mock_search, \
         patch("reranker_hf.rerank", return_value=[]):

        rag_tasks.retrieve("some question", k=5)


