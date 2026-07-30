import pytest
from unittest.mock import patch
import rag_tasks
import reranker_hf
import graph_builder
import main

from langchain_core.messages import HumanMessage, AIMessage


def test_retrieve_node_happy_path():
    state = {"question": "What is Homer Simpson's favorite food?"}
    result = graph_builder.retrieve_node(state)

    assert "retrieved_chunks" in result


def test_retrieve_node_unit():
    state = {"question": "What is Homer Simpson's favorite food?"}
    fake_chunks = [("chunk B", 0.91), ("chunk A", 0.72), ("chunk C", 0.55)]

    with patch("graph_builder.rag_tasks.retrieve", return_value=fake_chunks):
        result = graph_builder.retrieve_node(state)

    assert result["retrieved_chunks"] == fake_chunks


def test_retrieve_node_retrieve_failure():
    state = {"question": "What is Homer Simpson's favorite food?"}

    with patch(
        "graph_builder.rag_tasks.retrieve", side_effect=RuntimeError("model timeout")
    ):
        with pytest.raises(RuntimeError):
            graph_builder.retrieve_node(state)


def test_rerank_node_happy_path():
    state = {
        "question": "What is Homer Simpson's favorite food?",
        "retrieved_chunks": [("chunk B", 0.91), ("chunk A", 0.72), ("chunk C", 0.55)],
    }
    fake_reranked = [("chunk B", 0.91), ("chunk A", 0.72), ("chunk C", 0.55)]

    with patch("graph_builder.rag_tasks.rerank", return_value=fake_reranked):
        result = graph_builder.rerank_node(state)

    assert result["reranked_chunks"] == fake_reranked
    assert result["score"] == 0.91


def test_rerank_node_rerank_failure():
    state = {
        "question": "What is Homer Simpson's favorite food?",
        "retrieved_chunks": [("chunk B", 0.91), ("chunk A", 0.72)],
    }

    with patch(
        "graph_builder.rag_tasks.rerank", side_effect=RuntimeError("model timeout")
    ):
        with pytest.raises(RuntimeError):
            graph_builder.rerank_node(state)


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
        "reranked_chunks": [
            ("Homer loves donuts.", 0.9),
            ("He works at the plant.", 0.5),
        ],
        "history": [],
    }

    with patch(
        "main.generate_with_llm_failover",
        return_value="Homer's favorite food is donuts.",
    ) as mock_failover:
        result = await graph_builder.generate_with_context_node(state)
    assert result["reply"] == "Homer's favorite food is donuts."
    mock_failover.assert_called_once()


@pytest.mark.asyncio
async def test_generate_without_context_node_happy_path():
    state = {
        "question": "Who are Homer's family?",
        "retrieved_chunks": [],
        "history": [],
    }

    with patch(
        "main.generate_with_llm_failover",
        return_value="Homer's family includes Marge, Bart, Lisa, and Maggie.",
    ) as mock_failover:
        result = await graph_builder.generate_without_context_node(state)

    assert result["reply"] == "Homer's family includes Marge, Bart, Lisa, and Maggie."
    mock_failover.assert_called_once()


@pytest.mark.asyncio
async def test_generate_without_context_node_no_knowledge():
    state = {
        "question": "What is Homer's cholesterol level in the Season 12 finale?",
        "retrieved_chunks": [],
        "history": [],
    }

    with patch(
        "main.generate_with_llm_failover",
        return_value="NO_KNOWLEDGE",
    ):
        result = await graph_builder.generate_without_context_node(state)

    assert result["reply"] == "NO_KNOWLEDGE"


@pytest.mark.asyncio
async def test_web_search_node_happy_path():
    state = {
        "question": "What is the weather like today?",
        "history": [],
    }

    with patch(
        "web_search_provider.web_search_fallback",
        return_value="Today's weather is sunny with a high of 75F.",
    ), patch(
        "main.generate_with_llm_failover",
        return_value="It's sunny with a high of 75F today.",
    ):
        result = await graph_builder.web_search_node(state)
    assert result["reply"] == (
        "It's sunny with a high of 75F today.\n\n"
        "(Note: answer sourced from live web search, not local knowledge base.)"
    )


@pytest.mark.asyncio
async def test_web_search_node_no_results():
    state = {
        "question": "asdkfjhalskdjfh gibberish query",
        "history": [],
    }

    with patch(
        "web_search_provider.web_search_fallback",
        return_value=None,
    ):
        result = await graph_builder.web_search_node(state)

    assert result == {
        "reply": "I don't know - no local context, no trained knowledge, and web search returned nothing."
    }
