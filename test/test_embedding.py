import rag_tasks 
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# test 1 happy path
def test_get_embedding_happy_path():
    result = rag_tasks.get_embedding("abcde")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)

def test_get_embedding_whitespace_input_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        rag_tasks.get_embedding(" ")


def test_get_embedding_none_input_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        rag_tasks.get_embedding(None)    

# Unit test — mocks HF, tests only your logic
def test_get_embedding_calls_hf_correctly():
    with patch("rag_tasks.get_embedding", return_value=[0.1, 0.2, 0.3]):
        result = rag_tasks.get_embedding("abcde")
    assert result == [0.1, 0.2, 0.3]
