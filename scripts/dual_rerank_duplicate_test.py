import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.text_rag import embeddings_hf, vectorstore_chroma
from app.text_rag.graph_builder import bm25_retriever

test_queries = [
    "Homer Simpson",
    "Where is Homer now?",
    "Petronella",
    "which character is introverted",
    "the paraffin's gone soft again",
    "Bart Simpson",
    "who is Doreen",
    "Springfield Nuclear Power Plant",
    "Mervyn's middle name",
    "who created the Simpsons",
]

for q in test_queries:
    q_emb = embeddings_hf.get_embedding(q)
    chroma_results = vectorstore_chroma.search(query_embedding=q_emb, k=5)
    bm25_docs = bm25_retriever.invoke(q)
    bm25_results = [doc.page_content for doc in bm25_docs]

    overlap = set(chroma_results) & set(bm25_results)
    print(f"Query: {q!r} — overlap: {len(overlap)}/5")
