import pytest
from unittest.mock import patch
from graph_builder import retrieve_and_rerank_node
import rag_tasks
import reranker_hf

def test_retrieve_and_rerank_node_happy_path():
    state = {"question": "What is Homer Simpson's favorite food?"}
    result = retrieve_and_rerank_node(state)

    assert "retrieved_chunks" in result
    assert "score" in result
    assert result["score"] >= 0.0

def test_retrieve_and_rerank_node_unit():
    state = {"question": "What is Homer Simpson's favorite food?"}
    fake_reranked = [("chunk B", 0.91), ("chunk A", 0.72), ("chunk C", 0.55)]
    
    with patch("graph_builder.rag_tasks.retrieve", return_value=fake_reranked):
        result = retrieve_and_rerank_node(state)

    assert result["score"] == 0.91
    assert result["retrieved_chunks"] == fake_reranked
    
def test_retrieve_and_rerank_node_rerank_failure():
    state = {"question": "What is Homer Simpson's favorite food?"}

    with patch("graph_builder.rag_tasks.retrieve", side_effect=RuntimeError("model timeout")):
        with pytest.raises(RuntimeError):
            retrieve_and_rerank_node(state)