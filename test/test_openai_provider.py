import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
import openai_provider


@pytest.mark.asyncio
async def test_generate_answer_openai_happy_path():
    mock_response = MagicMock()
    mock_response.content = "Paris is the capital of France."

    with patch(
        "openai_provider.openai_llm.ainvoke",
        new=AsyncMock(return_value=mock_response),
    ) as mock_generate:
        reply = await openai_provider.generate_answer_openai(
            prompt="What is the capital of France?",
            history=[HumanMessage(content="hi")],
        )

    assert reply == "Paris is the capital of France."
    mock_generate.assert_called_once()

    # Verify messages were passed as LangChain message objects
    call_args = mock_generate.call_args.args
    messages = call_args[0]
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "What is the capital of France?"


@pytest.mark.asyncio
async def test_generate_answer_openai_retries_then_succeeds():
    mock_response = MagicMock()
    mock_response.content = "Recovered answer."

    with patch(
        "openai_provider.openai_llm.ainvoke",
        new=AsyncMock(side_effect=[Exception("temporary error"), mock_response]),
    ) as mock_generate: