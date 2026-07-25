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


@pytest.mark.asyncio
@patch("main.web_search_fallback", new_callable=AsyncMock)
@patch("main.generate_with_llm_failover", new_callable=AsyncMock)
async def test_generate_with_knowledge_failover_falls_back_to_web_search(
    mock_generate, mock_web_search
):
    # First call returns NO_KNOWLEDGE, second call (post-web-search) returns the real answer
    mock_generate.side_effect = [
        "NO_KNOWLEDGE",
        "Bart Simpson is Homer's son, according to web search.",
    ]
    mock_web_search.return_value = "some web search results"

    result = await main.generate_with_knowledge_failover(
        "who is his son", "promptstr", []
    )

    assert "web search" in result.lower()
    assert "(Note: answer sourced from live web search" in result


@pytest.mark.asyncio
@patch("main.web_search_fallback", new_callable=AsyncMock)
@patch("main.generate_with_llm_failover", new_callable=AsyncMock)
async def test_generate_with_knowledge_failover_error_path_no_web_results(
    mock_generate, mock_web_search
):
    mock_generate.return_value = "NO_KNOWLEDGE"
    mock_web_search.return_value = ""

    result = await main.generate_with_knowledge_failover(
        "some obscure question", "promptstr", []
    )

    assert isinstance(result, str)
    assert "no local context" in result.lower() or "don't know" in result.lower()


@patch("main.web_search_fallback", new_callable=AsyncMock)
@patch("main.generate_with_llm_failover", new_callable=AsyncMock)
def test_generate_with_knowledge_failover_edge_case_whitespace_around_sentinel(
    mock_generate, mock_web_search
):
    # edge case: the model's reply IS "NO_KNOWLEDGE", but with extra
    # whitespace/newlines around it — a boundary case of otherwise-valid
    # input. The code uses reply.strip() == "NO_KNOWLEDGE", so this
    # should still correctly trigger the web search fallback.
    mock_generate.side_effect = ["  NO_KNOWLEDGE  \n", "Grounded answer from the web."]
    mock_web_search.return_value = "some web search result text"

    result = asyncio.run(
        main.generate_with_knowledge_failover("some question", "promptstr", [])
    )

    assert isinstance(result, str)
    assert "Grounded answer from the web." in result


@patch("main.generate_with_llm_failover", new_callable=AsyncMock)
def test_generate_with_knowledge_failover_passes_history_through(mock_generate):
    mock_generate.return_value = "some answer"
    prior_history = [
        HumanMessage(content="earlier question"),
        AIMessage(content="earlier answer"),
    ]

    asyncio.run(
        main.generate_with_knowledge_failover(
            "new question", "promptstr", prior_history
        )
    )

    # confirm history was included in the messages sent to generate_with_llm_failover
    call_args = mock_generate.call_args
    messages_sent = call_args[0][1]  # second positional arg: messages_override
    assert prior_history[0] in messages_sent
    assert prior_history[1] in messages_sent


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
