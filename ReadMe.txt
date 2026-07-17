## Branches
- `main` — baseline RAG pipeline: OpenAI embeddings (`text-embedding-3-small`), FAISS retrieval, OpenAI generation.
- `feature/reranker-huggingface-gemini` — upgraded pipeline:
  - HuggingFace open-source embeddings (`all-MiniLM-L6-v2`), replacing OpenAI embeddings — removes a paid API dependency for the embedding step.
  - Two-stage retrieval: FAISS retrieves top-20 candidates, then a HuggingFace cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks down to the top-3.
  - Gemini 3 Flash for generation, with retry/failover logic.
  - Demonstrates provider-agnostic LLM integration and a retrieve-wide-then-rerank-narrow retrieval design.
- `feature/chromadb-migration` — current branch, adds Corrective RAG (CRAG):
  - Swaps FAISS for **ChromaDB** (`vectorstore_chroma.py`) as the vector store — persistent, no manual index rebuild required on restart.
  - **Two-tier failover, decoupled from each other:**
    1. *Network failover* (`generate_with_network_failover`): tries OpenAI first, falls back to Gemini on any exception (rate limit, timeout, outage).
    2. *Knowledge failover* (`generate_with_knowledge_failover`): if the model responds with the sentinel `NO_KNOWLEDGE` (meaning local context AND the model's own trained knowledge both came up empty), falls back to a live Tavily web search, then re-generates a grounded answer from those results — routed through the same network-failover call, so a provider outage during the correction step is still covered.
  - Prompt rules extracted into `prompt_rules.py` (`CONTEXT_ONLY_RULE`, `CONTEXT_TRAINED_DATA_ONLY_RULE`) so `main.py` isn't cluttered with long prompt strings, and the rules are independently testable/tunable.
  - Answers sourced from the web-search fallback are labeled in the reply (`"(Note: answer sourced from live web search, not local knowledge base.)"`) so it's clear when the pipeline had to leave the local knowledge base.

## Architecture (feature/chromadb-migration)
```
question
  → embed (HuggingFace all-MiniLM-L6-v2)
  → ChromaDB search (top-20)
  → cross-encoder rerank (top-3)
  → construct_prompt(CONTEXT_TRAINED_DATA_ONLY_RULE, ...)
  → generate_with_knowledge_failover
        → generate_with_network_failover (OpenAI → Gemini on failure)
        → if reply == "NO_KNOWLEDGE":
              → Tavily web search
              → construct_prompt(CONTEXT_ONLY_RULE, ...) on web results
              → generate_with_network_failover again (OpenAI → Gemini on failure)
  → reply
```

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

### Environment variables required
| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `main.py`, primary generation |
| `GEMINI_API_KEY` | `gemini_provider.py`, failover generation |
| `TAVILY_API_KEY` | `web_search_provider.py`, CRAG web-search fallback |
| `INPUT_FILE` | `rag_tasks.py`, source document to chunk/embed at ingest time |

## Timing breakdown (sample run, feature/reranker-huggingface-gemini)
| Stage | Time |
|---|---|
| Chunking | 0.00s |
| HuggingFace embedding | 0.12s |
| FAISS search | 0.00s |
| Cross-encoder rerank | 1.04s |
| Gemini generation | 9.30s |

Retrieval (embedding → FAISS → rerank) totals ~1.05s; generation dominates total latency at ~9x the retrieval cost. Within retrieval, cross-encoder reranking — not FAISS search — is the bottleneck, since it requires a full forward pass per candidate rather than a precomputed vector lookup.

*(Timing breakdown for the ChromaDB + CRAG path not yet re-benchmarked — swap FAISS row for ChromaDB search, and add a row for the Tavily fallback path when the NO_KNOWLEDGE branch fires.)*

## Testing
Full pytest suite under `test/`, ~99% coverage, one file per module:
- `test_chunking.py`, `test_retrieval.py`, `test_reranker.py`, `test_embedding.py` — core RAG pipeline pieces
- `test_main.py` — `construct_prompt`, `generate_with_network_failover`, `generate_with_knowledge_failover`
- `test_api.py` — `/chat` and `/health` endpoints, including provider-failure and validation error paths

Each function is tested for happy path, error path (a real failure — bad input, a dependency throwing), and edge case (valid but boundary input — empty strings, whitespace, etc.) where all three are meaningfully different from each other.

All external calls (OpenAI, Gemini, Tavily, ChromaDB) are mocked — the suite runs with no live API calls and no network dependency.

Run tests:
```bash
pytest test/ -v
```

Run with coverage:
```bash
pytest test/ --cov=. --cov-report=term-missing --cov-config=.coveragerc
```
(`.coveragerc` excludes manual/diagnostic scripts — `selenium_testing.py`, `test_gemini_speed.py`, `time_test_reranker_warmup.py` — that aren't part of the pytest suite.)

## Deployment
Deployed via Docker to both **Render** and **Google Cloud Run**.

Known deployment gotchas worth documenting (all resolved):
- **Cloud Run + memory:** the ML stack (sentence-transformers, cross-encoder, FAISS, ChromaDB) needs real headroom — 512Mi caused a silent out-of-memory kill on container startup with no traceback in the logs. Bumped to 2Gi.
- **Cloud Run + HTTPS scheme:** Cloud Run terminates TLS at its proxy, so the container only ever sees plain `http` internally. FastAPI's `url_for()` generated `http://` links for static assets, which browsers block as mixed content on an otherwise-`https://` page. Fixed by adding `--proxy-headers --forwarded-allow-ips='*'` to the uvicorn CMD, so it trusts Cloud Run's `X-Forwarded-Proto` header instead of guessing.
- **Windows PowerShell + git branch typos:** watch for truncated branch names in push commands (`feature/chromadb` vs `feature/chromadb-migration`).
