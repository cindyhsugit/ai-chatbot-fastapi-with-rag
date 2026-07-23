import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import StateGraph

import graph_builder


def test_node_wired_into_graph():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("retrieve_and_rerank", graph_builder.retrieve_and_rerank_node)
    graph.set_entry_point("retrieve_and_rerank")
    graph.set_finish_point("retrieve_and_rerank")
    compiled = graph.compile()

    result = compiled.invoke({"question": "what is the refund policy?"})

    assert "score" in result


@pytest.mark.asyncio
async def test_graph_routes_low_score_to_web_search():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("retrieve_and_rerank", graph_builder.retrieve_and_rerank_node)
    graph.add_node("generate_with_context", graph_builder.generate_with_context_node)
    graph.add_node(
        "generate_without_context", graph_builder.generate_without_context_node
    )
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        graph_builder.score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.set_entry_point("retrieve_and_rerank")
    compiled = graph.compile()

    with patch(
        "graph_builder.rag_tasks.retrieve", return_value=[("irrelevant chunk", -5.0)]
    ), patch("main.generate_with_knowledge_failover", return_value="Some answer"):
        result = await compiled.ainvoke({"question": "some question", "history": []})

    assert result["reply"] == "Some answer"


@pytest.mark.asyncio
async def test_generate_with_context_node_wired_into_graph():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("generate_with_context", graph_builder.generate_with_context_node)
    graph.set_entry_point("generate_with_context")
    graph.set_finish_point("generate_with_context")
    compiled = graph.compile()
    initial_state = {
        "question": "What is Homer Simpson's favorite food?",
        "retrieved_chunks": [("Homer loves donuts.", 0.9)],
        "history": [],
    }

    with patch(
        "main.generate_with_network_failover",
        return_value="Homer's favorite food is donuts.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's favorite food is donuts."


@pytest.mark.asyncio
async def test_generate_without_context_node_wired_into_graph():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node(
        "generate_without_context", graph_builder.generate_without_context_node
    )
    graph.set_entry_point("generate_without_context")
    graph.set_finish_point("generate_without_context")
    compiled = graph.compile()

    initial_state = {
        "question": "Who are Homer's family?",
        "retrieved_chunks": [],
        "history": [],
    }

    with patch(
        "main.generate_with_knowledge_failover",
        return_value="Homer's family includes Marge, Bart, Lisa, and Maggie.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's family includes Marge, Bart, Lisa, and Maggie."


@pytest.mark.asyncio
async def test_full_graph_routes_to_web_search_on_no_knowledge():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("retrieve_and_rerank", graph_builder.retrieve_and_rerank_node)
    graph.add_node("generate_with_context", graph_builder.generate_with_context_node)
    graph.add_node(
        "generate_without_context", graph_builder.generate_without_context_node
    )
    graph.add_node("web_search_node", graph_builder.web_search_node)

    graph.set_entry_point("retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        graph_builder.score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.add_conditional_edges(
        "generate_without_context",
        graph_builder.not_in_context_router,
        {
            "web_search_node": "web_search_node",
            "END": graph_builder.END,
        },
    )
    graph.add_edge("generate_with_context", graph_builder.END)
    graph.add_edge("web_search_node", graph_builder.END)

    compiled = graph.compile()

    initial_state = {
        "question": "What is Homer's cholesterol level in the Season 12 finale?",
        "history": [],
    }

    with patch(
        "graph_builder.rag_tasks.retrieve",
        return_value=[
            ("irrelevant chunk", -5.0)
        ],  # low score -> generate_without_context
    ), patch(
        "main.generate_with_knowledge_failover",
        return_value="NO_KNOWLEDGE",  # triggers web search
    ), patch(
        "web_search_provider.web_search_fallback",
        return_value="Some fact found via web search.",
    ), patch(
        "main.generate_with_network_failover",
        return_value="Homer's cholesterol level was mentioned as high in that episode.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert (
        result["reply"]
        == "Homer's cholesterol level was mentioned as high in that episode."
    )


@pytest.mark.asyncio
async def test_full_graph_end_to_end_with_context():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("retrieve_and_rerank", graph_builder.retrieve_and_rerank_node)
    graph.add_node("generate_with_context", graph_builder.generate_with_context_node)
    graph.add_node(
        "generate_without_context", graph_builder.generate_without_context_node
    )
    graph.add_node("web_search_node", graph_builder.web_search_node)

    graph.set_entry_point("retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        graph_builder.score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.add_conditional_edges(
        "generate_without_context",
        graph_builder.not_in_context_router,
        {
            "web_search_node": "web_search_node",
            "END": graph_builder.END,
        },
    )
    graph.add_edge("generate_with_context", graph_builder.END)
    graph.add_edge("web_search_node", graph_builder.END)

    compiled = graph.compile()

    initial_state = {
        "question": "What is Homer Simpson's favorite food?",
        "history": [],
    }

    with patch(
        "graph_builder.rag_tasks.retrieve",
        return_value=[("Homer loves donuts.", 0.5)],
    ), patch(
        "main.generate_with_network_failover",
        return_value="Homer's favorite food is donuts.",
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's favorite food is donuts."


@pytest.mark.asyncio
async def test_full_graph_end_to_end_without_context():
    graph = StateGraph(graph_builder.ChatState)
    graph.add_node("retrieve_and_rerank", graph_builder.retrieve_and_rerank_node)
    graph.add_node("generate_with_context", graph_builder.generate_with_context_node)
    graph.add_node("generate_without_context", graph_builder.generate_without_context_node)
    graph.add_node("web_search_node", graph_builder.web_search_node)

    graph.set_entry_point("retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        graph_builder.score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.add_conditional_edges(
        "generate_without_context",
        graph_builder.not_in_context_router,
        {
            "web_search_node": "web_search_node",
            "END": graph_builder.END,
        },
    )
    graph.add_edge("generate_with_context", graph_builder.END)
    graph.add_edge("web_search_node", graph_builder.END)

    compiled = graph.compile()

    initial_state = {
        "question": "Who are Homer's family?",
        "history": [],
    }

    with patch(
        "graph_builder.rag_tasks.retrieve",
        return_value=[
            ("irrelevant chunk", -5.0)
        ],  # low score -> generate_without_context
    ), patch(
        "main.generate_with_knowledge_failover",
        return_value="Homer's family includes Marge, Bart, Lisa, and Maggie.",  # not NO_KNOWLEDGE
    ):
        result = await compiled.ainvoke(initial_state)

    assert result["reply"] == "Homer's family includes Marge, Bart, Lisa, and Maggie."
