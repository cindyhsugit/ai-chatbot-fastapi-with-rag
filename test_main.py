from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
import main

client = TestClient(main.app)

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
    

def test_chatAPIEndpointTestDirect():
    request = main.ChatRequest(message="hello")
    response = main.chatAPIEndpoint(request)
    
    assert "reply" in response
    assert isinstance(response["reply"], str)
    assert response["reply"].strip() != ""


from unittest.mock import patch, MagicMock
def test_chatAPIEndpointTestDirect_patch():
    request = main.ChatRequest(message="hello")

    mock_openai_response = MagicMock()
    mock_openai_response.choices = [
        MagicMock(message=MagicMock(content="fake reply"))
    ]

    with patch("main.client.chat.completions.create", return_value=mock_openai_response):
        response = main.chatAPIEndpoint(request)

    assert response == {"reply": "fake reply"}