from unittest.mock import patch, MagicMock
import asyncio
import pytest
import app.providers.web_search_provider as web_search_provider
from unittest.mock import AsyncMock
from app.text_rag.graph_builder import web_search_node
import app.text_rag.prompt_rules as prompt_rules
from langchain_core.messages import HumanMessage, AIMessage


@patch("app.providers.web_search_provider.tavily_client")
def test_web_search_fallback_happy_path(mock_tavily):
    # happy path: Tavily returns real results with title + content
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "Homer Simpson - Wikipedia",
                "content": "Broccoli casserole is a running joke.",
            }
        ]
    }

    result = asyncio.run(
        web_search_provider.web_search_fallback("Homer Simpson favorite food")
    )

    assert isinstance(result, str)
    assert "Broccoli casserole is a running joke." in result
    assert "Homer Simpson - Wikipedia" in result  # source label included


@patch("app.providers.web_search_provider.tavily_client")
def test_web_search_fallback_error_path_tavily_throws(mock_tavily):
    # error path: Tavily itself raises (network error, bad API key,
    # rate limit) — fails soft, returns "" instead of crashing the pipeline
    mock_tavily.search.side_effect = Exception("Tavily API down")

    result = asyncio.run(web_search_provider.web_search_fallback("any question"))

    assert result == ""


# Covers line 48: the happy-path and empty-content tests always supply at least
# one entry in "results", and the exception test never reaches line 46. This
# test mocks Tavily returning a successful response with no usable results
# (empty list or missing key), so web_search_fallback returns "" early.
@pytest.mark.parametrize(
    "tavily_response",
    [
        {"results": []},
        {},
    ],
)
@patch("app.providers.web_search_provider.tavily_client")
def test_web_search_fallback_empty_results_returns_empty_string(
    mock_tavily, tavily_response
):
    mock_tavily.search.return_value = tavily_response

    result = asyncio.run(web_search_provider.web_search_fallback("any question"))

    assert result == ""


@patch("app.providers.web_search_provider.tavily_client")
def test_web_search_fallback_edge_case_results_with_empty_content(mock_tavily):
    # edge case: Tavily succeeds and returns a result, but its "content"
    # field is blank/whitespace-only — should be skipped, not included
    # as an empty snippet
    mock_tavily.search.return_value = {
        "results": [
            {"title": "Some Page", "content": "   "},
            {"title": "Real Result", "content": "Actual useful content."},
        ]
    }

    result = asyncio.run(web_search_provider.web_search_fallback("any question"))

    assert "Real Result" in result
    assert "Actual useful content." in result
    assert "Some Page" not in result  # blank-content result correctly filtered out


@pytest.mark.asyncio
async def test_web_search_node_uses_web_search_rule():
    state = {"question": "what is trump's necktie color today", "history": []}

    with (
        patch(
            "app.providers.web_search_provider.web_search_fallback",
            new=AsyncMock(return_value="some search result text"),
        ),
        patch(
            "app.main.generate_with_llm_failover",
            new=AsyncMock(return_value="a synthesized answer"),
        ) as mock_generate,
    ):
        result = await web_search_node(state)

        assert mock_generate.await_count == 1
        prompt = mock_generate.await_args.kwargs["prompt"]
        assert prompt_rules.WEB_SEARCH_RULE in prompt
        assert "some search result text" in prompt

        expected_reply = "a synthesized answer\n\n(Note: answer sourced from live web search, not local knowledge base.)"

        assert result["reply"] == expected_reply
        assert result["history"] == [
            HumanMessage(content=state["question"]),
            AIMessage(content=expected_reply),
        ]


@pytest.mark.asyncio
async def test_web_search_node_no_results_returns_fallback_message():
    state = {"question": "some obscure question", "history": []}

    with patch(
        "app.providers.web_search_provider.web_search_fallback",
        new=AsyncMock(return_value=None),
    ):
        result = await web_search_node(state)

        assert result["reply"] == (
            "I don't know - no local context, no trained knowledge, and web search returned nothing."
        )
        assert "history" not in result
