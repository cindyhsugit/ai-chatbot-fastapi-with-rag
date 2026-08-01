# Open ai and gemini response is different
# format
# OpenAI — response.content is a plain string
# Gemini — response.content can be a list of dicts


# open AI expect "role" of "user" or "assistant" in message
# input
# history = [
#     HumanMessage(content="What's the capital of France?"),
#     AIMessage(content="The capital of France is Paris."),
#     HumanMessage(content="What's its population?"),
#     AIMessage(content="Paris has a population of about 2.1 million people."),
# ]
# output
# [
#     {"role": "user", "content": "What's the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris."},
#     {"role": "user", "content": "What's its population?"},
#     {"role": "assistant", "content": "Paris has a population of about 2.1 million people."},
# ]
def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)
