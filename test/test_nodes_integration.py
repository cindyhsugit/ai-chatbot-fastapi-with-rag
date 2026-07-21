import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import StateGraph

from graph_builder import ChatState, retrieve_and_rerank_node
from graph_builder import score_threshold_router, generate_with_context_node, generate_without_context_node

def test_node_wired_into_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.set_entry_point("retrieve_and_rerank")
    graph.set_finish_point("retrieve_and_rerank")
    compiled = graph.compile()

    result = compiled.invoke({"question": "what is the refund policy?"})

    assert "score" in result

@pytest.mark.asyncio
async def test_graph_routes_low_score_to_web_search():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.set_entry_point("retrieve_and_rerank")
    compiled = graph.compile()

    with patch("graph_builder.rag_tasks.retrieve", return_value=[("irrelevant chunk", -5.0)]), \
         patch("main.generate_with_knowledge_failover", return_value="Some answer"):
        result = await compiled.ainvoke({"question": "some question", "history": []})

    assert result["reply"] == "Some answer"

@pytest.mark.asyncio
async def test_generate_with_context_node_wired_into_graph():
    graph = StateGraph(ChatState)
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.set_entry_point("generate_with_context")
    graph.set_finish_point("generate_with_context")
    compiled = graph.compile()
    initial_state = {    
        "question": "What is Homer Simpson's favorite food?",
        "retrieved_chunks": [("Homer loves donuts.", 0.9)],
        "history": []
    }

    with patch(
        "main.generate_with_network_failover",
        return_value="Homer's favorite food is donuts.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's favorite food is donuts."

@pytest.mark.asyncio
async def test_generate_without_context_node_wired_into_graph():
    graph = StateGraph(ChatState)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.set_entry_point("generate_without_context")
    graph.set_finish_point("generate_without_context")
    compiled = graph.compile()

    initial_state = {
        "question": "Who are Homer's family?",
        "retrieved_chunks": [],
        "history": []
    }

    with patch(
        "main.generate_with_knowledge_failover",
        return_value="Homer's family includes Marge, Bart, Lisa, and Maggie.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's family includes Marge, Bart, Lisa, and Maggie."