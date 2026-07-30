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
