# Open ai and gemini response is different
# format
# OpenAI — response.content is a plain string
# Gemini — response.content can be a list of dicts
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
