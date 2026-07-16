from rag_tasks import chunk_text

# test 1 happy path
def test_chunk_text_happy_path():
   assert(chunk_text("abcde") == ["abcde"])

# test 1 error path
def test_chunk_text_error_path():
    assert(chunk_text(" ") == [])

# test 1 edge case 
def test_chunk_text_edge_case():
    assert(chunk_text("") == [])
