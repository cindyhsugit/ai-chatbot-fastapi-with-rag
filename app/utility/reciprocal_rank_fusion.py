from collections import defaultdict
from typing import List


def combine(*retriever_pools: List[str], rrf_k: int = 60, top_n: int = 10) -> List[str]:
    """
    Combines multiple lists of text chunks using Reciprocal Rank Fusion (RRF).

    Args:
        *retriever_pools: Variable number of lists, each containing retrieved text chunks (list[str]).
        rrf_k: Smoothing constant for RRF formula (default: 60).
        top_n: Maximum number of top ranked chunks to return.

    Returns:
        List[str]: Merged and re-ranked list of top N text chunks.
    """
    rrf_scores = defaultdict(float)

    for pool in retriever_pools:
        for rank, text in enumerate(pool, start=1):
            rrf_scores[text] += 1.0 / (rrf_k + rank)

    # Sort chunks by accumulated RRF score descending
    sorted_chunks = sorted(
        rrf_scores.keys(), key=lambda text: rrf_scores[text], reverse=True
    )

    return sorted_chunks[:top_n]
