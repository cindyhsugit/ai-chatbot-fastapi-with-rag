from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest
from main import app

client = TestClient(app)

def test_healthAPIEndpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
# using TestClient    
def test_chatAPIEndpointTestClient():
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert data["reply"] != ""
    
@patch("main.async_client.chat.completions.create", new_callable=AsyncMock)
@patch("main.rag_tasks.retrieve_async", new_callable=AsyncMock)
def test_chat_returns_reply(mock_retrieve, mock_create):
    mock_retrieve.return_value = ["Test context chunk."]
    mock_create.return_value.choices = [
        type("Choice", (), {"message": type("Msg", (), {"content": "Test reply"})()})
    ]
    response = client.post("/chat", json={"message": "What is this about?"})
    assert response.status_code == 200
    assert response.json()["reply"] == "Test reply"