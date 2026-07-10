from sentence_transformers import SentenceTransformer
import numpy as np

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