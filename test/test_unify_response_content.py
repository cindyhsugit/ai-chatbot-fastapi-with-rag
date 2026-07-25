import utility.unify_response_content


def test_to_text_plain_string():
    # OpenAI-style response: content is already a plain string
    result = utility.unify_response_content.to_text("Paris is the capital of France.")
    assert result == "Paris is the capital of France."


def test_to_text_list_of_dicts_gemini_shape():
    # Gemini-style response: content is a list of dicts with a "text" key
    content = [
        {
            "type": "text",
            "text": "Homer Simpson's favorite food is broccoli casserole.",
            "extras": {"signature": "abc123"},
        }
    ]
    result = utility.unify_response_content.to_text(content)
    assert result == "Homer Simpson's favorite food is broccoli casserole."


def test_to_text_list_of_multiple_dicts():
    # Multiple text blocks should be joined together
    content = [
        {"type": "text", "text": "Part one."},
        {"type": "text", "text": "Part two."},
    ]
    result = utility.unify_response_content.to_text(content)
    assert result == "Part one. Part two."


def test_to_text_list_of_plain_strings():
    # Edge case: list containing plain strings instead of dicts
    content = ["Hello", "world"]
    result = utility.unify_response_content.to_text(content)
    assert result == "Hello world"


def test_to_text_empty_list():
    # Edge case: empty list should return empty string, not crash
    result = utility.unify_response_content.to_text([])
    assert result == ""


def test_to_text_dict_without_text_key():
    # Edge case: dict missing the "text" key falls back to str(block)
    content = [{"type": "unknown", "value": 42}]
    result = utility.unify_response_content.to_text(content)
    assert "42" in result


def test_to_text_non_str_non_list_input():
    # Edge case: unexpected type (e.g. None or int) should stringify gracefully
    result = utility.unify_response_content.to_text(None)
    assert result == "None"

    result = utility.unify_response_content.to_text(123)
    assert result == "123"
