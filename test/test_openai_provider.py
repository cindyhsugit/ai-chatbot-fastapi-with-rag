import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
import providers.openai_provider
from langchain_openai import ChatOpenAI


@pytest.mark.asyncio
async def test_generate_answer_openai_happy_path():
    mock_response = MagicMock()
    mock_response.content = "Paris is the capital of France."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("providers.openai_provider.openai_llm", mock_llm):
        reply = await providers.openai_provider.generate_answer_openai(
            prompt="What is the capital of France?",
            history=[HumanMessage(content="hi")],
        )

    assert reply == "Paris is the capital of France."
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_openai_retries_exhausted_raises():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("persistent failure"))

    with patch("providers.openai_provider.openai_llm", mock_llm):
        with pytest.raises(Exception, match="persistent failure"):
            await providers.openai_provider.generate_answer_openai(
                prompt="Always fails", history=[]
            )

    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_openai_edge_case_succeeds_on_last_retry():
    mock_response = MagicMock()
    mock_response.content = "Recovered on final attempt."

    with patch.object(
        ChatOpenAI,
        "ainvoke",
        new=AsyncMock(
            side_effect=[
                Exception("transient error 1"),
                Exception("transient error 2"),
                mock_response,
            ]
        ),
    ) as mock_ainvoke:
        reply = await providers.openai_provider.generate_answer_openai(
            prompt="Recovers eventually", history=[]
        )

    assert reply == "Recovered on final attempt."
    assert mock_ainvoke.call_count == 3
