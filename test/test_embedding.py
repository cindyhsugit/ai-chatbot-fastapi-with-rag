from rag_tasks import get_embedding
import numpy as np
import pytest

# test 1 happy path
def test_get_embedding_happy_path():
    result = get_embedding("abcde")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)

def test_get_embedding_whitespace_input_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        get_embedding(" ")


def test_get_embedding_none_input_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        get_embedding(None)    