from langgraph.graph import StateGraph
from graph_builder import ChatState, retrieve_and_rerank_node

def test_node_wired_into_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.set_entry_point("retrieve_and_rerank")
    graph.set_finish_point("retrieve_and_rerank")
    compiled = graph.compile()

    result = compiled.invoke({"question": "what is the refund policy?"})

    assert "score" in result