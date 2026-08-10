import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import utility


def split(loaded_text: str) -> list[str]:
    """
    Initializes the SemanticChunker, processes the text, filters empty chunks
    using utility, and returns a list of raw string chunks with timing.
    """
    text_splitter = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        breakpoint_threshold_type="standard_deviation",
    )

    start = time.time()
    chunks = text_splitter.create_documents([loaded_text])

    # Call the reusable utility function
    chunks = utility.chunks_utils.filter_empty_chunks(chunks)

    chunk_texts = [c.page_content for c in chunks]
    end = time.time()

    print(f"Chunk text Time: {end-start:.2f}s")
    return chunk_texts
