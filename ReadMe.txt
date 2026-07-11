## Branches
- `main` — baseline RAG pipeline: OpenAI embeddings (`text-embedding-3-small`), FAISS retrieval, OpenAI generation.
- `feature/reranker-huggingface-gemini` — upgraded pipeline:
  - HuggingFace open-source embeddings (`all-MiniLM-L6-v2`), replacing OpenAI embeddings — removes a paid API dependency for the embedding step.
  - Two-stage retrieval: FAISS retrieves top-20 candidates, then a HuggingFace cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks down to the top-3.
  - Gemini 3 Flash for generation, with retry/failover logic.
  - Demonstrates provider-agnostic LLM integration and a retrieve-wide-then-rerank-narrow retrieval design.

## Running it
Three ways to query the pipeline:
1. In Python: `rag_tasks.retrieve("your question")`, then `python main.py`
2. FastAPI's built-in `/docs` endpoint — manually construct a request in the interactive UI
3. curl:
```bash
   curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "your question"}'
```

## Timing breakdown (sample run)
| Stage | Time |
|---|---|
| Chunking | 0.00s |
| HuggingFace embedding | 0.12s |
| FAISS search | 0.00s |
| Cross-encoder rerank | 1.04s |
| Gemini generation | 9.30s |

Retrieval (embedding → FAISS → rerank) totals ~1.05s; generation dominates total latency at ~9x the retrieval cost. Within retrieval, cross-encoder reranking — not FAISS search — is the bottleneck, since it requires a full forward pass per candidate rather than a precomputed vector lookup.
