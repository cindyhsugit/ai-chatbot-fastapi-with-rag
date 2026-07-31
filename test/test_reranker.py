from reranker_hf import rerank
import pytest
import rag_tasks
import torch
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np


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
    top_text, top_score = result[0]
    assert "broccoli casserole" in top_text


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


def test_rag_tasks_rerank_wrapper_calls_hf_rerank_and_returns_result():
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "The weather in Springfield is often sunny.",
        "Homer Simpson's favorite food is broccoli casserole.",
    ]
    expected = [("Homer Simpson's favorite food is broccoli casserole.", 0.9)]

    with patch("rag_tasks.reranker_hf.rerank", return_value=expected) as mock_rerank:
        result = rag_tasks.rerank(query, candidates, top_k=2)

    assert result == expected
    mock_rerank.assert_called_once_with(query, candidates, top_k=3)


def test_rerank_with_onnx_returns_top_result_first_for_tensor_logits():
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "The weather in Springfield is often sunny.",
        "Homer Simpson's favorite food is broccoli casserole.",
    ]

    class DummyTokenizer:
        def __call__(self, pairs, padding=True, truncation=True, return_tensors="pt"):
            return {
                "input_ids": [[1, 2], [3, 4]],
                "attention_mask": [[1, 1], [1, 1]],
            }

    class DummyOnnxModel:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[0.2], [0.8]]))

    with patch.object(rag_tasks, "tokenizer", DummyTokenizer()), patch.object(
        rag_tasks, "onnx_model", DummyOnnxModel()
    ):
        result = rag_tasks.rerank_with_onnx(query, candidates, top_k=2)

    assert len(result) == 2
    assert result[0][0] == "Homer Simpson's favorite food is broccoli casserole."
    assert result[0][1] == pytest.approx(0.8)
    assert result[1][0] == "The weather in Springfield is often sunny."
    assert result[1][1] == pytest.approx(0.2)


def test_rerank_with_onnx_numpy_scores_fallback_raises():
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "The weather in Springfield is often sunny.",
        "Homer Simpson's favorite food is broccoli casserole.",
    ]

    class DummyTokenizer:
        def __call__(self, pairs, padding=True, truncation=True, return_tensors="pt"):
            return {
                "input_ids": [[1, 2], [3, 4]],
                "attention_mask": [[1, 1], [1, 1]],
            }

    class DummyOnnxModel:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=np.array([[0.2], [0.8]]))

    with patch.object(rag_tasks, "tokenizer", DummyTokenizer()), patch.object(
        rag_tasks, "onnx_model", DummyOnnxModel()
    ):
        with pytest.raises(ValueError, match="only convert an array of size 1"):
            rag_tasks.rerank_with_onnx(query, candidates, top_k=1)
