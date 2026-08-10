# Standard library
import os
import time
from pathlib import Path

# Third-party
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Local modules
import app.text_rag.embeddings_hf as embeddings_hf
import app.config as config
from app.text_rag import vectorstore_chroma
import app.text_rag.semantic_chunk as semantic_chunk
import app.utility.file_io

# Setup
load_dotenv()
load_dotenv("apiKey.env")


# Retrieve relevant chunks for a new question


def retrieve(question: str, k: int = 20) -> list[tuple[str, float]]:
    # question_embedding -> [0.0119, -0.0440, 0.0801, ...]   (1536 floats, similar to chunk 0's embedding)
    question_embedding = embeddings_hf.get_embedding(question)

    start = time.time()

    retrieved_chunks = vectorstore_chroma.search(
        query_embedding=question_embedding, k=k
    )

    end = time.time()
    print(f"-- Chroma DB search Time: {end-start:.2f}s")

    return retrieved_chunks


if __name__ == "__main__":
    print(
        "This file builds a search index when imported — run main.py instead of this file directly."
    )
else:
    # Loading
    filepath = os.getenv("INPUT_FILE")
    loaded_text = app.utility.file_io.safely_open_input_file(filepath)

    # Chunking
    chunks = semantic_chunk.split(loaded_text)

    # Embedding
    embeddings = embeddings_hf.embed_chunks(chunks)

    # Indexing
    vectorstore_chroma.add_documents(embeddings, chunks)
