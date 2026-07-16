from reranker_hf import rerank
import pytest


def test_rerank_happy_path():
    # happy path: a genuinely relevant candidate should score higher
    # and end up ranked first
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "The weather in Springfield is often sunny.",
        "Homer Simpson's favorite food is broccoli casserole.",
    ]
    result = rerank(query, candidates)

    assert isinstance(result, list)
    assert len(result) > 0
    assert "broccoli casserole" in result[0]


def test_rerank_error_path_empty_candidates():
    # error path: no candidates to rank at all — should return an
    # empty list, not crash
    result = rerank("any question", [])
    assert result == []


def test_rerank_edge_case_all_irrelevant_candidates():
    # edge case: candidates exist, but none are actually relevant to
    # the query. Without a score_threshold, rerank() still returns
    # top_k candidates regardless of relevance — this documents that
    # current behavior (not a bug, just how it works without a cutoff).
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "Stock prices rose sharply today.",
        "The capital of France is Paris.",
    ]
    result = rerank(query, candidates, top_k=2)

    assert isinstance(result, list)
    assert len(result) == 2  # returns both, even though neither is relevant