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
        result = await main.generate_with_llm_failover("promptstr")

    assert result == "hello back"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_with_llm_failover_openai_fails_uses_gemini():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("all providers down"))

    with patch("main.backup_llm", mock_llm):
        with pytest.raises(Exception, match="all providers down"):
            await main.generate_with_llm_failover("promptstr")


def test_construct_prompt_happy_path():
    # happy path: normal rules, context, and question all get placed correctly
    result = main.construct_prompt(
        prompt_rules.CONTEXT_ONLY_RULE,
        "Homer Simpson's favorite food is broccoli casserole.",
        "What is Homer's favorite food?",
    )
    assert isinstance(result, str)
    assert prompt_rules.CONTEXT_ONLY_RULE in result
    assert "Homer Simpson's favorite food is broccoli casserole." in result
    assert "What is Homer's favorite food?" in result
    # confirms the labels are actually present, not just the raw text
    assert "Context:" in result
    assert "Question:" in result


def test_construct_prompt_error_path_none_does_not_raise():
    # error path: unlike chunk_text(None), this does NOT raise an
    # exception — f-strings call str() on their arguments instead of
    # calling methods like .split() on them. So None silently becomes
    # the literal text "None" embedded in the prompt sent to the model.
    # This documents that (mis)behavior rather than a crash — worth
    # knowing, since a silent bug (model sees "Context:\nNone") is
    # arguably worse than a loud crash, because nothing tells you it
    # happened.
    result = main.construct_prompt(prompt_rules.CONTEXT_ONLY_RULE, None, "a question")
    assert isinstance(result, str)
    assert "None" in result  # documents the current (unintended?) behavior


def test_construct_prompt_edge_case_empty_context_and_question():
    # edge case: empty strings are valid input, not a crash — but the
    # resulting prompt still needs to be well-formed enough for the
    # model to receive it without confusion
    result = main.construct_prompt(prompt_rules.CONTEXT_ONLY_RULE, "", "")
    assert isinstance(result, str)
    assert "Context:" in result
    assert "Question:" in result
