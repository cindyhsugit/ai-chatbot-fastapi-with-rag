## Branches

- `main` — baseline RAG pipeline: OpenAI embeddings (`text-embedding-3-small`), FAISS retrieval, OpenAI generation.
- [`feature/reranker-huggingface-gemini`](https://github.com/cindyhsugit/ai-chatbot-fastapi-with-rag/tree/feature/reranker-huggingface-gemini) — upgraded pipeline: HuggingFace open-source embeddings (`all-MiniLM-L6-v2`), a cross-encoder reranking stage (`ms-marco-MiniLM-L-6-v2`) on top of FAISS retrieval, and Gemini as an alternate/added generation provider. Demonstrates a two-stage retrieval architecture (retrieve wide, rerank narrow) and provider-agnostic LLM integration.
