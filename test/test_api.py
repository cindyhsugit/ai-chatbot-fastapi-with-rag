from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
# using TestClient    
def test_chat_endpoint_happy_path():
    response = client.post("/chat", json={"message": "hello",  "session_id": "test-session-1"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert data["reply"] != ""

def test_chat_endpoint_error_path():
    """Error path: request body is missing the required 'message' field entirely."""
    response = client.post("/chat", json={})
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_chat_endpoint_wrong_type_for_message():
    """Error path: 'message' is present but the wrong type."""
    response = client.post("/chat", json={"message": 12345})
    assert response.status_code == 422

@patch("main.async_client.chat.completions.create", new_callable=AsyncMock)
@patch("main.gemini_provider.generate_answer_gemini", new_callable=AsyncMock)
def test_chat_endpoint_both_providers_fail(mock_gemini, mock_openai):
    """Error path: OpenAI fails AND the Gemini failover also fails."""
    mock_openai.side_effect = Exception("OpenAI down")
    mock_gemini.side_effect = Exception("Gemini down too")

    # raise_server_exceptions=False makes TestClient behave like a real
    # deployed server would — converting an unhandled exception into an
    # actual 500 response instead of re-raising it into the test itself
    client_no_raise = TestClient(app, raise_server_exceptions=False)
    response = client_no_raise.post("/chat", json={"message": "hello", "session_id": "test-session-1"})

    assert response.status_code == 500

import requests

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