from sentence_transformers import CrossEncoder

# Loaded once at startup
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Takes the user query and a list of candidate chunk texts (from FAISS retrieval),
# returns the top_k chunks reordered by relevance.
def rerank(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    
    # query -> "what's homer's favorite food" (str)
    # candidates -> ["chunk 1 text...", "chunk 2 text...", ...] (list[str])
    #
    # pairs -> [["what's homer's favorite food", "chunk 1 text..."],
    #           ["what's homer's favorite food", "chunk 2 text..."], ...]
    # each pair feeds the cross-encoder query+doc together
    
    pairs = [[query, candidate] for candidate in candidates]
    scores = model.predict(pairs)

    # Sort candidates by score, descending
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [text for text, score in scored[:top_k]]