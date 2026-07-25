from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import pytest

import providers.gemini_provider as gemini_provider


@pytest.mark.asyncio
async def test_generate_answer_gemini_happy_path():
    mock_response = MagicMock()
    mock_response.text = "Paris is the capital of France."

    with patch(
        "gemini_provider.gemini_llm.ainvoke",
        new=AsyncMock(return_value=mock_response),
    ) as mock_generate:
        reply = await gemini_provider.generate_answer_gemini(
            prompt="What is the capital of France?",
            history=[HumanMessage(content="hi")],
        )

    assert reply == "Paris is the capital of France."
    mock_generate.assert_called_once()

    # Verify contents was converted to Gemini's role/parts shape, not raw dicts
    call_kwargs = mock_generate.call_args
    messages = call_args[0]
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "What is the capital of France?"


@pytest.mark.asyncio
async def test_generate_answer_gemini_error_path_retries_then_raises():
    with patch(
        "gemini_provider.gemini_llm.ainvoke",
        new=AsyncMock(side_effect=Exception("503 Service Unavailable")),
    ) as mock_generate, patch(
        "gemini_provider.asyncio.sleep", new=AsyncMock()
    ) as mock_sleep:

        with pytest.raises(Exception, match="503 Service Unavailable"):
            await gemini_provider.generate_answer_gemini(
                prompt="What is the capital of France?",
                history=[],
                max_retries=3,
            )

    assert mock_generate.call_count == 3
    assert mock_sleep.call_count == 2  # sleeps between attempts, not after the last


@pytest.mark.asyncio
async def test_generate_answer_gemini_edge_case_succeeds_on_last_retry():
    mock_response = MagicMock()
    mock_response.text = "Paris is the capital of France."

    with patch(
        "gemini_provider.gemini_llm.ainvoke",
        new=AsyncMock(
            side_effect=[
                Exception("503 Service Unavailable"),
                Exception("503 Service Unavailable"),
                mock_response,
            ]
        ),
    ) as mock_generate, patch(
        "gemini_provider.asyncio.sleep", new=AsyncMock()
    ) as mock_sleep:

        reply = await gemini_provider.generate_answer_gemini(
            prompt="What is the capital of France?",
            history=[],
            max_retries=3,
        )

    assert reply == "Paris is the capital of France."
    assert mock_generate.call_count == 3
    assert (
        mock_sleep.call_count == 2
    )  # backoff before attempt 2 and attempt 3, none after success
