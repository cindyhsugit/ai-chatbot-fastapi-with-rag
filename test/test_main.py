from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import main
import prompt_rules
import asyncio
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_generate_with_llm_failover_happy_path():
    mock_response = MagicMock()
    mock_response.content = "hello back"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    with patch("main.backup_llm", mock_llm):
        result = await main.generate_with_llm_failover("promptstr", messages=[])

    assert result == "hello back"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_with_llm_failover_openai_fails_uses_gemini():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("all providers down"))

    with patch("main.backup_llm", mock_llm):
        with pytest.raises(Exception, match="all providers down"):
            await main.generate_with_llm_failover("promptstr", messages=[])
