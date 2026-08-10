from langchain_core.documents import Document


def filter_empty_chunks(chunks: list[Document]) -> list[Document]:
    """
    Drops empty/whitespace-only chunks the semantic chunker can produce
    at document boundaries or around blank lines in the source text.
    Prints a warning if any were dropped, so silent data loss stays visible.
    """
    original_count = len(chunks)
    filtered = [c for c in chunks if c.page_content and c.page_content.strip()]
    dropped = original_count - len(filtered)
    if dropped:
        print(
            f"Warning: dropped {dropped} empty/whitespace-only chunk(s) after semantic chunking"
        )
    return filtered
