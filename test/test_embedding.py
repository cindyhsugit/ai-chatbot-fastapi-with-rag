from rag_tasks import get_embedding
import numpy as np
import pytest

# test 1 happy path
def test_get_embedding_happy_path():
   assert(isinstance(get_embedding("abcde"), np.ndarray))

# test 1 error path
def test_get_embedding_error_path():
    assert(isinstance(get_embedding(" "), np.ndarray))


# test 1 edge case 
def test_get_embedding_edge_case():
    with pytest.raises(ValueError):
        get_embedding(None)