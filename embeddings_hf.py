from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Loaded once at startup, not per-request
model = SentenceTransformer("all-MiniLM-L6-v2")

# input looks like
# texts = ["hello world", "what is FAISS", "local RAG search"]
# output looks like
# array([[0.12, 0.34, 0.56, ...],
#        [0.22, 0.18, 0.91, ...],
#        [0.44, 0.51, 0.27, ...]], dtype=float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Takes a list of strings, returns embeddings as a numpy array (float32, required by FAISS)."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")


def get_embedding(text: str) -> list[float]:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return_list = embed_texts([text])[0].tolist()

    return return_list


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Iterates over chunks, generates embeddings using get_embedding,
    measures execution time, and validates dimensions.
    """
    embeddings = []
    start = time.time()

    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    end = time.time()

    if embeddings:
        print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"hugging face embedding Time: {end-start:.2f}s")

    return embeddings
