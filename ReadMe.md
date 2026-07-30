## Architecture Diagram

![LangGraph flow](screenshots/graph_diagram_v3.png)

## Demo Trace

![RAG workflow demo](screenshots/combined_demo_annotated.png)

See the [high-level architecture](screenshots/diagram_hierarchy.md) and
[detailed structure](screenshots/diagram_detail.md) for more.

# AI Chatbot with RAG — LangGraph Edition

A retrieval-augmented chatbot built from primitives, then migrated to
[LangGraph](https://github.com/langchain-ai/langgraph) for orchestration,
conversational memory, and conditional routing between retrieval, trained
knowledge, and live web search.

** [Cloud Run deployment] Live demo:** [ai-chatbot-with-rag-1016078012439.us-west1.run.app](https://ai-chatbot-with-rag-1016078012439.us-west1.run.app)

> Looking for the original, from-scratch implementation (no LangChain/LangGraph)?
> See [README_v1_manual_rag.md](README_v1_manual_rag.md) — kept live on Render
> as a before/after comparison.

---

## Architecture

This app implements a three-tier CRAG (Corrective RAG) pipeline as a LangGraph
`StateGraph`:

```
question
   │
   ▼
retrieve_node  (ChromaDB vector search, top-20)
   │
   ▼
rerank_node  (cross-encoder reranker, top-3 + score)
   │
   ├─ score ≥ threshold ──► generate_with_context_node ──► reply
   │
   └─ score < threshold ──► generate_without_context_node
                                  │
                                  ├─ confident answer ──► reply
                                  │
                                  └─ NO_KNOWLEDGE ──► web_search_node ──► reply
```

- **Retrieval:** HuggingFace embeddings (`all-MiniLM-L6-v2`) + ChromaDB vector
  store (`retrieve_node`)
- **Reranking:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) narrows top-20
  retrieved chunks down to the top 3 most relevant (`rerank_node`) — split
  into its own node so LangSmith traces retrieval and reranking latency
  separately
- **Generation:** Two-tier LLM failover — OpenAI primary, Gemini fallback, via
  LangChain's `with_fallbacks()`
- **Fallback:** Tavily web search when neither local context nor trained
  knowledge can answer the question
- **Memory:** LangGraph's `MemorySaver` checkpointer + `add_messages` reducer
  give the graph multi-turn conversational memory, keyed by `session_id`

## Why LangGraph

The original version (see `README_v1_manual_rag.md`) was built entirely from
primitives — no LangChain — specifically to understand RAG internals deeply
before reaching for a framework. This version migrates orchestration to
LangGraph to get:

- **Explicit state management** — a typed `ChatState` schema instead of
  threading dicts through function calls by hand
- **Conditional routing** — graph edges replace nested if/else fallback logic
- **Built-in conversational memory** — `MemorySaver` + `add_messages` replaced
  a hand-rolled `session_store` dict, removing a redundant/competing source of
  truth for conversation history
- **Provider abstraction** — `ChatOpenAI` / `ChatGoogleGenerativeAI` replaced
  custom message-format converters for OpenAI and Gemini

## Notable debugging story

**Combined retrieve+rerank node obscured the real bottleneck.** LangSmith
traces initially showed one node's total latency without distinguishing
vector search from cross-encoder reranking. Split the original
`retrieve_and_rerank_node` into `retrieve_node` and `rerank_node` so each
step traces separately — confirmed the cross-encoder reranking step, not the
vector search, was the dominant cost, and opened the door to targeted
optimizations (fewer reranked candidates, lower-precision inference, smaller
model variants) without touching retrieval at all.

**Dual history tracking.** Early in the migration, conversation history was
tracked in two places at once: a hand-rolled `session_store` dict, and
LangGraph's `MemorySaver` checkpointer (via the `add_messages` reducer). Both
were being fed into the graph on every turn — a redundancy that risked
silently duplicating messages as conversations grew. Caught this during
architecture review and removed `session_store` entirely, making the
checkpointer the single source of truth for history.

**Cross-provider response shape mismatch.** After migrating to LangChain's
`ChatOpenAI` / `ChatGoogleGenerativeAI` wrappers with `with_fallbacks()`, a
Gemini fallback response rendered as `[object Object]` in the UI. Traced this
to a shape mismatch: OpenAI's `response.content` is always a plain string,
while Gemini's can be a list of content blocks (`[{"type": "text", "text":
"...", "extras": {...}}]`) carrying internal metadata. Wrote a small
normalization utility (`unify_response_content.to_text`) so downstream code
never needs to know which provider actually answered.

