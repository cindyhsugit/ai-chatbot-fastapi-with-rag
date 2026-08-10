"""
web_search_provider.py

Fallback web search provider for the CRAG (Corrective RAG) path.
Used only when:
  1. Local context (ChromaDB) has no answer, AND
  2. The LLM's own trained knowledge returns NO_KNOWLEDGE

Design choice: no reranking on the returned snippets. This is a rare
fallback-of-a-fallback path (context miss + trained-data miss), so we
optimize for simplicity/speed over precision. Tavily already returns
clean, LLM-oriented snippets (not raw HTML), so an extra filter pass
adds cost without much benefit here.
"""

import os
from tavily import TavilyClient

# Reads API key from environment - never hardcode it
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


async def web_search_fallback(query: str, max_results: int = 3) -> str:
    """
    Runs a web search and returns concatenated snippet text,
    ready to drop directly into an LLM prompt as context.

    No embedding, no reranking - see module docstring for rationale.

    Returns:
        A single string of concatenated snippets, or an empty string
        if no results were found (caller should handle that case,
        e.g. by falling back to a polite "I don't know").
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # "advanced" costs more credits; basic is enough for snippet-only use
        )
    except Exception as e:
        # Fail soft: treat search failure like "no results", don't crash the pipeline
        print(f"[web_search_fallback] Tavily search failed: {e}")
        return ""

    results = response.get("results", [])
    if not results:
        return ""

    # Concatenate top snippets with light source labeling.
    # Keeps it simple - no relevance filtering, matches your "skip rerank" decision.
    snippets = []
    for r in results:
        title = r.get("title", "Untitled")
        content = r.get("content", "").strip()
        if content:
            snippets.append(f"[Source: {title}]\n{content}")

    return "\n\n".join(snippets)
