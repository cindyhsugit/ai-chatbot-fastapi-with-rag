from unittest.mock import patch, AsyncMock
import asyncio
import pytest
from gemini_provider import generate_answer_gemini


@patch("gemini_provider.client.aio.models.generate_content", new_callable=AsyncMock)
def test_generate_answer_gemini_happy_path(mock_generate_content):
    # happy path: Gemini responds successfully on the first try
    mock_generate_content.return_value = type("Response", (), {"text": "Broccoli casserole."})()

    result = asyncio.run(generate_answer_gemini("What is Homer's favorite food?"))

    assert isinstance(result, str)
    assert result == "Broccoli casserole."
    assert mock_generate_content.call_count == 1  # succeeded first try, no retries needed


@patch("gemini_provider.client.aio.models.generate_content", new_callable=AsyncMock)
def test_generate_answer_gemini_error_path_retries_then_raises(mock_generate_content):
    # error path: every attempt fails — confirms it retries the
    # correct number of times, then gives up loudly instead of
    # hanging forever or silently swallowing the failure
    mock_generate_content.side_effect = Exception("Gemini API down")

    with pytest.raises(Exception, match="Gemini API down"):
        asyncio.run(generate_answer_gemini("some question", max_retries=3))

    assert mock_generate_content.call_count == 3


@patch("gemini_provider.client.aio.models.generate_content", new_callable=AsyncMock)
def test_generate_answer_gemini_edge_case_succeeds_on_last_retry(mock_generate_content):
    # edge case: fails twice, succeeds on the final allowed attempt —
    # confirms the retry loop doesn't give up early on a transient
    # failure that would have recovered
    mock_generate_content.side_effect = [
        Exception("timeout"),
        Exception("timeout again"),
        type("Response", (), {"text": "Broccoli casserole."})(),
    ]

    result = asyncio.run(generate_answer_gemini("some question", max_retries=3))

    assert result == "Broccoli casserole."
    assert mock_generate_content.call_count == 3