from unittest.mock import patch, MagicMock
import asyncio
import pytest
from web_search_provider import web_search_fallback


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