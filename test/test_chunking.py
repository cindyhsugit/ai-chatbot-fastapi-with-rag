from rag_tasks import chunk_text

# test 1 happy path
def test_chunk_text_happy_path():
   assert(chunk_text("abcde") == ["abcde"])

# test error path
def test_chunk_text_error_path():
    assert(chunk_text(" ") == [])

def test_chunk_text_exact_multiple_of_chunk_size():
    # edge case: word count divides evenly, no leftover partial chunk
    text = " ".join(f"w{i}" for i in range(6))
    result = chunk_text(text, chunk_size=3)
    assert len(result) == 2
    
# test edge case 
def test_chunk_text_edge_case():
    assert(chunk_text("") == [])

def test_chunk_text_empty_string():
    # edge case: valid input, but the boundary case of "nothing at all"
    assert chunk_text("") == []
