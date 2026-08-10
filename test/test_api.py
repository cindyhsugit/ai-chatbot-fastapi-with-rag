from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest
from app.main import app
import requests
from langchain_core.messages import HumanMessage

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Covers line 97: the home route's TemplateResponse return. Existing API tests
# hit /health and /langgraphchat but never GET /, so home() was never invoked.
def test_home_endpoint_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "The Simpsons RAG AI" in response.text


# Covers line 104: the /langgraph route's TemplateResponse return. Same template
# as /, but this is a separate async handler — no existing test requested it.
def test_langgraph_page_endpoint_returns_html():
    response = client.get("/langgraph")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "The Simpsons RAG AI" in response.text


# # using TestClient
# def test_chat_endpoint_happy_path():
#     response = client.post(
#         "/chat", json={"message": "hello", "session_id": "test-session-1"}
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert "reply" in data
#     assert isinstance(data["reply"], str)
#     assert data["reply"] != ""


# def test_chat_endpoint_error_path():
#     """Error path: request body is missing the required 'message' field entirely."""
#     response = client.post("/chat", json={})
#     assert response.status_code == 422  # FastAPI/Pydantic validation error


# def test_chat_endpoint_wrong_type_for_message():
#     """Error path: 'message' is present but the wrong type."""
#     response = client.post("/chat", json={"message": 12345})
#     assert response.status_code == 422


# @patch("main.async_client.chat.completions.create", new_callable=AsyncMock)
# @patch("providers.gemini_provider.generate_answer_gemini", new_callable=AsyncMock)
# def test_chat_endpoint_both_providers_fail(mock_gemini, mock_openai):
#     """Error path: OpenAI fails AND the Gemini failover also fails."""
#     mock_openai.side_effect = Exception("OpenAI down")
#     mock_gemini.side_effect = Exception("Gemini down too")

#     # raise_server_exceptions=False makes TestClient behave like a real
#     # deployed server would — converting an unhandled exception into an
#     # actual 500 response instead of re-raising it into the test itself
#     client_no_raise = TestClient(app, raise_server_exceptions=False)
#     response = client_no_raise.post(
#         "/chat", json={"message": "hello", "session_id": "test-session-1"}
#     )

#     assert response.status_code == 500


def test_langgraphchat_endpoint_happy_path():
    response = client.post(
        "/langgraphchat",
        json={"message": "what is the refund policy?", "session_id": "test-session-1"},
    )
    assert response.status_code == 200
    assert "reply" in response.json()


def test_langgraphchat_endpoint_missing_message_field():
    response = client.post(
        "/langgraphchat",
        json={"session_id": "test-session-1"},  # missing required "message" field
    )
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_langgraphchat_endpoint_empty_message_returns_400():
    response = client.post(
        "/langgraphchat",
        json={"message": "   ", "session_id": "test-session-1"},  # whitespace-only
    )
    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


@pytest.mark.asyncio
@patch("main.graph.ainvoke", new_callable=AsyncMock)
async def test_langgraphchat_homer_favorite_food(mock_ainvoke):
    mock_ainvoke.return_value = {
        "reply": "Based on the provided context, Homer Simpson's favorite food is broccoli casserole.",
        "history": [],
        "question": "What is Homer Simpson's favorite food?",
        "session_id": "test-session-1",
        "retrieved_chunks": [],
        "score": 0.9,
    }

    response = client.post(
        "/langgraphchat",
        json={
            "message": "What is Homer Simpson's favorite food?",
            "session_id": "test-session-1",
        },
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "broccoli casserole" in reply


def test_langgraphchat_homer_favorite_food_integration():
    """Integration test — hits real graph, real LLM, real fabricated-fact grounding check."""
    response = client.post(
        "/langgraphchat",
        json={
            "message": "What is Homer Simpson's favorite food?",
            "session_id": "test-session-integration",
        },
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "broccoli casserole" in reply
