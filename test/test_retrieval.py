from rag_tasks import retrieve
import numpy as np
import pytest

# test 1 happy path
def test_retrieve_happy_path():
   assert(isinstance(retrieve("a question here"), list))

# test 1 error path
def test_retrieve_error_path():
    assert(isinstance(retrieve("12345"), list))

# test 1 edge case 
def test_retrieve_edge_case():
    assert(isinstance(retrieve(" "), list))
