from unittest.mock import patch, MagicMock
import asyncio
import pytest
from web_search_provider import web_search_fallback
from unittest.mock import AsyncMock
from graph_builder import web_search_node
import prompt_rules

@patch("web_search_provider.tavily_client")
def test_web_search_fallback_happy_path(mock_tavily):
    # happy path: Tavily returns real results with title + content
    mock_tavily.search.return_value = {
        "results": [
            {"title": "Homer Simpson - Wikipedia", "content": "Broccoli casserole is a running joke."}
        ]
    }

    result = asyncio.run(web_search_fallback("Homer Simpson favorite food"))

    assert isinstance(result, str)
    assert "Broccoli casserole is a running joke." in result
    assert "Homer Simpson - Wikipedia" in result  # source label included


@patch("web_search_provider.tavily_client")
def test_web_search_fallback_error_path_tavily_throws(mock_tavily):
    # error path: Tavily itself raises (network error, bad API key,
    # rate limit) — fails soft, returns "" instead of crashing the pipeline
    mock_tavily.search.side_effect = Exception("Tavily API down")

    result = asyncio.run(web_search_fallback("any question"))

    assert result == ""


@patch("web_search_provider.tavily_client")
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

    result = asyncio.run(web_search_fallback("any question"))

    assert "Real Result" in result
    assert "Actual useful content." in result
    assert "Some Page" not in result  # blank-content result correctly filtered out

@pytest.mark.asyncio
async def test_web_search_node_uses_web_search_rule():
    state = {"question": "what is trump's necktie color today", "history": []}

    with patch("graph_builder.web_search_provider.web_search_fallback", new=AsyncMock(return_value="some search result text")), \
         patch("main.construct_prompt") as mock_construct_prompt, \
         patch("main.generate_with_llm_failover", new=AsyncMock(return_value="a synthesized answer")):

        mock_construct_prompt.return_value = "built prompt"

        result = await web_search_node(state)

        # Assert the WEB_SEARCH_RULE was used, not CONTEXT_ONLY_RULE
        called_rule = mock_construct_prompt.call_args.args[0] if mock_construct_prompt.call_args.args else mock_construct_prompt.call_args.kwargs["rules"]
        assert called_rule == prompt_rules.WEB_SEARCH_RULE

        assert result["reply"] == "a synthesized answer"
        assert result["history"] == [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": "a synthesized answer"},
        ]


@pytest.mark.asyncio
async def test_web_search_node_no_results_returns_fallback_message():
    state = {"question": "some obscure question", "history": []}

    with patch("graph_builder.web_search_provider.web_search_fallback", new=AsyncMock(return_value=None)):
        result = await web_search_node(state)

        assert result["reply"] == (
            "I don't know - no local context, no trained knowledge, and web search returned nothing."
        )
        assert "history" not in result