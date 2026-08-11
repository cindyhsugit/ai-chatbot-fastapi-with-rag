import app.text_rag.reranker_hf as reranker_hf
import pytest
import app.text_rag.rag_tasks as rag_tasks
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
    result = reranker_hf.rerank(query, candidates)

    assert isinstance(result, list)
    assert len(result) > 0
    top_text, top_score = result[0]
    assert "broccoli casserole" in top_text


def test_rerank_error_path_empty_candidates():
    # error path: no candidates to rank at all — should return an
    # empty list, not crash
    result = reranker_hf.rerank("any question", [])
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
    result = reranker_hf.rerank(query, candidates, top_k=2)

    assert isinstance(result, list)
    assert len(result) == 2  # returns both, even though neither is relevant


def test_calculate_max_length_uses_chunk_size_overlap_and_query_margin():
    class DummyTokenizer:
        def encode(self, text):
            return text.split()

    sample_text = "The quick brown fox jumps over the lazy dog. " * 15
    sample_tokens = len(sample_text.split())
    chars_per_token = len(sample_text) / sample_tokens
    expected = int(((len(sample_text) / chars_per_token) + 20) * 1.15)

    result = reranker_hf.calculate_max_length(
        sample_text,
        DummyTokenizer(),
    )

    assert result == expected


def test_rerank_with_onnx_returns_top_result_first_for_tensor_logits():
    query = "What is Homer Simpson's favorite food?"
    candidates = [
        "The weather in Springfield is often sunny.",
        "Homer Simpson's favorite food is broccoli casserole.",
    ]

    class DummyTokenizer:
        def encode(self, text):
            return text.split()

        def __call__(
            self,
            pairs,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors="pt",
        ):
            return {
                "input_ids": SimpleNamespace(shape=(2, 2)),
                "attention_mask": SimpleNamespace(shape=(2, 2)),
            }

    class DummyOnnxModel:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[0.2], [0.8]]))

    with (
        patch.object(reranker_hf, "tokenizer", DummyTokenizer()),
        patch.object(reranker_hf, "onnx_model", DummyOnnxModel()),
    ):
        result = reranker_hf.rerank_with_onnx(query, candidates, top_k=2)

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
        def encode(self, text):
            return text.split()

        def __call__(
            self,
            pairs,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors="pt",
        ):
            return {
                "input_ids": SimpleNamespace(shape=(2, 2)),
                "attention_mask": SimpleNamespace(shape=(2, 2)),
            }

    class DummyOnnxModel:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=np.array([[0.2], [0.8]]))

    with (
        patch.object(reranker_hf, "tokenizer", DummyTokenizer()),
        patch.object(reranker_hf, "onnx_model", DummyOnnxModel()),
    ):
        with pytest.raises(ValueError, match="can only convert an array of size 1"):
            reranker_hf.rerank_with_onnx(query, candidates, top_k=1)
