## Branches

- `main` — baseline RAG pipeline: OpenAI embeddings (`text-embedding-3-small`), FAISS retrieval, OpenAI generation.
- [`feature/reranker-huggingface-gemini`](https://github.com/cindyhsugit/ai-chatbot-fastapi-with-rag/tree/feature/reranker-huggingface-gemini) — upgraded pipeline: HuggingFace open-source embeddings (`all-MiniLM-L6-v2`), a two-stage retrieval architecture with cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) on top of FAISS, and Gemini 3 Flash for generation. Demonstrates provider-agnostic LLM integration and retrieve-wide-then-rerank-narrow retrieval design.
