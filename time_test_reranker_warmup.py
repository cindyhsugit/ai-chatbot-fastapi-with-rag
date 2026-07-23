"""
Isolation test: does the cross-encoder get faster on its 2nd call,
regardless of which vector store fed it?

If call #2 is consistently faster than call #1, it's a warm-up effect
(model weights/ops already loaded in memory), not a FAISS vs ChromaDB difference.
"""

import time
from reranker_hf import rerank                  
import vectorstore_chroma as VectorStore_Chroma
from rag_tasks import get_embedding  # replace with wherever get_embedding lives

# Use the SAME question and SAME retrieved chunks for both calls,
# so the only variable is "which call number is this" (1st vs 2nd).
question = "what's homer's favorite food?"

# Pull real top-20 chunks once, reuse for both timed calls
question_embedding = get_embedding(question) # however you currently generate this embedding
retrieved_chunks = VectorStore_Chroma.search(
    query_embedding=question_embedding, k=20
)

print(f"Chunks retrieved: {len(retrieved_chunks)}")
print(f"Avg chunk length (chars): {sum(len(c) for c in retrieved_chunks) / len(retrieved_chunks):.0f}")

# --- Call 1 ---
start = time.time()
reranked_1 = rerank(question, retrieved_chunks, top_k=3)
time_1 = time.time() - start
print(f"Call 1 (cold?) time: {time_1:.2f}s")

# --- Call 2, identical input, same process ---
start = time.time()
reranked_2 = rerank(question, retrieved_chunks, top_k=3)
time_2 = time.time() - start
print(f"Call 2 (warm?) time: {time_2:.2f}s")

# --- Call 3, just to confirm the pattern holds ---
start = time.time()
reranked_3 = rerank(question, retrieved_chunks, top_k=3)
time_3 = time.time() - start
print(f"Call 3 (warm?) time: {time_3:.2f}s")

print()
print("If call 1 is much slower than calls 2 and 3, it's a warm-up effect")
print("(model loading/JIT on first use), not a FAISS-vs-ChromaDB difference.")