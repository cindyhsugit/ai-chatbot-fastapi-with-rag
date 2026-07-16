from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest
from main import generate_with_network_failover
import prompt_rules

@pytest.mark.anyio
async def test_generate_with_network_failover_happy_path():
    result = await generate_with_network_failover("hello", "promptstr")
    assert isinstance(result, str)
    assert result != ""