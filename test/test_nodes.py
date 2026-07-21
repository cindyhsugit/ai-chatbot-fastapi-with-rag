import pytest
from unittest.mock import patch
import rag_tasks
import reranker_hf
import graph_builder 
import main

def test_retrieve_and_rerank_node_happy_path():
    state = {"question": "What is Homer Simpson's favorite food?"}
    result = graph_builder.retrieve_and_rerank_node(state)

    assert "retrieved_chunks" in result
    assert "score" in result
    assert result["score"] >= 0.0

def test_retrieve_and_rerank_node_unit():
    state = {"question": "What is Homer Simpson's favorite food?"}
    fake_reranked = [("chunk B", 0.91), ("chunk A", 0.72), ("chunk C", 0.55)]
    
    with patch("graph_builder.rag_tasks.retrieve", return_value=fake_reranked):
        result = graph_builder.retrieve_and_rerank_node(state)

    assert result["score"] == 0.91
    assert result["retrieved_chunks"] == fake_reranked
    
def test_retrieve_and_rerank_node_rerank_failure():
    state = {"question": "What is Homer Simpson's favorite food?"}

    with patch("graph_builder.rag_tasks.retrieve", side_effect=RuntimeError("model timeout")):
        with pytest.raises(RuntimeError):
            graph_builder.retrieve_and_rerank_node(state)

def test_score_threshold_router_above_threshold():
    state = {"score": 0.5}
    assert graph_builder.score_threshold_router(state) == "generate_with_context"

def test_score_threshold_router_below_threshold():
    state = {"score": -3.0}
    assert graph_builder.score_threshold_router(state) == "generate_without_context"

def test_score_threshold_router_at_boundary():
    state = {"score": 0.0}
    assert graph_builder.score_threshold_router(state) == "generate_without_context"

from unittest.mock import patch
@pytest.mark.asyncio
async def test_generate_with_context_node_happy_path():
    state = {
        "question": "What is Homer Simpson's favorite food?",
        "retrieved_chunks": [("Homer loves donuts.", 0.9), ("He works at the plant.", 0.5)],
        "history": [],
    }

    with patch(
        "main.generate_with_network_failover",
        return_value="Homer's favorite food is donuts.",
    ) as mock_failover:
        result = await graph_builder.generate_with_context_node(state)

    assert result == {"reply": "Homer's favorite food is donuts."}
    mock_failover.assert_called_once()  