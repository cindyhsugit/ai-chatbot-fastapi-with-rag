import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utility import file_io, chunks_utils


def semantic_chunk(loaded_text: str) -> list[str]:
    """
    Initializes the SemanticChunker, processes the text, filters empty chunks
    using utility, and returns a list of raw string chunks with timing.
    """
    text_splitter = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        breakpoint_threshold_type="interquartile",
    )

    chunks = text_splitter.create_documents([loaded_text])

    # Call the reusable utility function
    chunks = chunks_utils.filter_empty_chunks(chunks)

    chunk_texts = [c.page_content for c in chunks]

    return chunk_texts


def fixed_size_chunk(
    chunks: list[str], max_chars: int = 800, overlap: int = 50
) -> list[str]:
    """
    Enforces a hard maximum size on each chunk. Chunks already under
    max_chars pass through unchanged; anything larger gets split further.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap
    )

    result = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            result.extend(splitter.split_text(chunk))
        else:
            result.append(chunk)
    return result


def get_sem_fs_chunk(loaded_text: str) -> list[str]:
    t0 = time.time()
    chunks = semantic_chunk(loaded_text)
    t1 = time.time()
    chunks = fixed_size_chunk(chunks, max_chars=800)
    t2 = time.time()
    print(f"semantic_chunk: {t1-t0:.2f}s, fixed_size_chunk: {t2-t1:.2f}s")

    return chunks
