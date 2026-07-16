from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest
from main import generate_with_network_failover
from main import generate_with_knowledge_failover
from main import construct_prompt
import prompt_rules
import asyncio

@patch("main.async_client.chat.completions.create", new_callable=AsyncMock)
def test_generate_with_network_failover_happy_path(mock_openai):
    mock_openai.return_value = type(
        "Response", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": "hello back"})()})]}
    )()

    result = asyncio.run(generate_with_network_failover("promptstr"))

    assert isinstance(result, str)
    assert result == "hello back"
    mock_openai.assert_called_once()

@patch("main.gemini_provider.generate_answer_gemini", new_callable=AsyncMock)
@patch("main.async_client.chat.completions.create", new_callable=AsyncMock)
def test_generate_with_network_failover_openai_fails_uses_gemini(mock_openai, mock_gemini):
    mock_openai.side_effect = Exception("OpenAI is down")
    mock_gemini.return_value = "Fallback answer from Gemini."

    result = asyncio.run(generate_with_network_failover("promptstr"))

    assert result == "Fallback answer from Gemini."
    mock_gemini.assert_called_once()  # confirms it actually fell back, not just returned something


@patch("main.generate_with_network_failover", new_callable=AsyncMock)
def test_generate_with_knowledge_failover_happy_path(mock_generate):
    # happy path: the model answers normally, no NO_KNOWLEDGE, no web search needed
    mock_generate.return_value = "Homer Simpson's favorite food is broccoli casserole."

    result = asyncio.run(
        generate_with_knowledge_failover("What is Homer's favorite food?", "promptstr")
    )

    assert isinstance(result, str)
    assert result != ""
    assert "broccoli casserole" in result


@patch("main.web_search_fallback", new_callable=AsyncMock)
@patch("main.generate_with_network_failover", new_callable=AsyncMock)
def test_generate_with_knowledge_failover_error_path_no_web_results(
    mock_generate, mock_web_search
):
    # error path: model says NO_KNOWLEDGE, and the web search fallback
    # comes back completely empty — does the code fail gracefully
    # instead of crashing or returning something broken?
    mock_generate.return_value = "NO_KNOWLEDGE"
    mock_web_search.return_value = ""  # no web results found

    result = asyncio.run(
        generate_with_knowledge_failover("some obscure question", "promptstr")
    )

    assert isinstance(result, str)
    assert "no local context" in result.lower() or "don't know" in result.lower()


@patch("main.web_search_fallback", new_callable=AsyncMock)
@patch("main.generate_with_network_failover", new_callable=AsyncMock)
def test_generate_with_knowledge_failover_edge_case_whitespace_around_sentinel(
    mock_generate, mock_web_search
):
    # edge case: the model's reply IS "NO_KNOWLEDGE", but with extra
    # whitespace/newlines around it — a boundary case of otherwise-valid
    # input. The code uses reply.strip() == "NO_KNOWLEDGE", so this
    # should still correctly trigger the web search fallback.
    mock_generate.side_effect = ["  NO_KNOWLEDGE  \n", "Grounded answer from the web."]
    mock_web_search.return_value = "some web search result text"

    result = asyncio.run(generate_with_knowledge_failover("some question", "promptstr"))

    assert isinstance(result, str)
    assert "Grounded answer from the web." in result


def test_construct_prompt_happy_path():
    # happy path: normal rules, context, and question all get placed correctly
    result = construct_prompt(
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
    result = construct_prompt(prompt_rules.CONTEXT_ONLY_RULE, None, "a question")
    assert isinstance(result, str)
    assert "None" in result  # documents the current (unintended?) behavior


def test_construct_prompt_edge_case_empty_context_and_question():
    # edge case: empty strings are valid input, not a crash — but the
    # resulting prompt still needs to be well-formed enough for the
    # model to receive it without confusion
    result = construct_prompt(prompt_rules.CONTEXT_ONLY_RULE, "", "")
    assert isinstance(result, str)
    assert "Context:" in result
    assert "Question:" in result
