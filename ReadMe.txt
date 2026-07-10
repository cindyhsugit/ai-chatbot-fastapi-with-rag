## Branches

- `main` — baseline RAG pipeline: OpenAI embeddings (`text-embedding-3-small`), FAISS retrieval, OpenAI generation.
-This branch has upgraded feature: HuggingFace open-source embeddings (`all-MiniLM-L6-v2`), a two-stage retrieval architecture
- Plus a cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) also from HuggingFace on top of FAISS, and Gemini 3 Flash for generation. Demonstrates provider-agnostic LLM integration and retrieve-wide-then-rerank-narrow retrieval design.
- Able to bypass external API calls for embeddings,  3 ways to run:
  rag_tasks.retrieve("Questions.....") then run python main.py
  fastAPI built in /doc endpoint manually create the request
  curl command "curl -X POST http://127.0.0.1:8000/chat -H 
                "Content-Type: application/json" -d '{\"message\": \"Question.....\"}'"
    
