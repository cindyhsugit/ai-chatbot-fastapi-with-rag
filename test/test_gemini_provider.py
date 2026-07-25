from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import pytest
from langchain_core.messages import HumanMessage
import providers.gemini_provider as gemini_provider


@pytest.mark.asyncio
async def test_generate_answer_gemini_happy_path():
    mock_response = MagicMock()
    mock_response.content = "Paris is the capital of France."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("providers.gemini_provider.gemini_llm", mock_llm):
        reply = await gemini_provider.generate_answer_gemini(
            prompt="What is the capital of France?",
            history=[HumanMessage(content="hi")],
        )

    assert reply == "Paris is the capital of France."
    mock_llm.ainvoke.assert_called_once()

    call_args = mock_llm.ainvoke.call_args[0]
    messages = call_args[0]
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "What is the capital of France?"


@pytest.mark.asyncio
async def test_generate_answer_gemini_retries_exhausted_raises():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("persistent failure"))

    with patch("providers.gemini_provider.gemini_llm", mock_llm):
        with pytest.raises(Exception, match="persistent failure"):
            await gemini_provider.generate_answer_gemini(
                prompt="Always fails", history=[]
            )

    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_gemini_edge_case_succeeds_on_last_retry():
    from langchain_google_genai import ChatGoogleGenerativeAI

    mock_response = MagicMock()
    mock_response.content = "Recovered on final attempt."

    with patch.object(
        ChatGoogleGenerativeAI,
        "ainvoke",
        new=AsyncMock(
            side_effect=[
                Exception("transient error 1"),
                Exception("transient error 2"),
                mock_response,  # succeeds on 3rd (last allowed) attempt
            ]
        ),
    ) as mock_ainvoke:
        reply = await gemini_provider.generate_answer_gemini(
            prompt="Recovers eventually", history=[]
        )

    assert reply == "Recovered on final attempt."
    assert mock_ainvoke.call_count == 3  # stop_after_attempt=3
