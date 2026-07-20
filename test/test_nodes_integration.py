from unittest.mock import patch
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

def test_graph_routes_low_score_to_web_search():
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

    with patch("graph_builder.rag_tasks.retrieve", return_value=[("irrelevant chunk", -5.0)]):
        result = compiled.invoke({"question": "some question"})

    # assert the graph actually reached generate_without_context's behavior    