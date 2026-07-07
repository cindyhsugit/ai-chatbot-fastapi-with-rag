from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once at startup, not per-request
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> np.ndarray:
    """Takes a list of strings, returns embeddings as a numpy array (float32, required by FAISS)."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")