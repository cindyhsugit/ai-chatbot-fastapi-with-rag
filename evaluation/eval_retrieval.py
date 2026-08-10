from langsmith import Client
from langsmith.evaluation import evaluate
from dotenv import load_dotenv
import asyncio
from openai import OpenAI

load_dotenv()
load_dotenv("apikey.env")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.text_rag.graph_builder import build_graph
from evaluation.datasets import get_or_create_dataset, SINGLE_TURN_EXAMPLES
from evaluation.evaluators import retrieval_relevance_evaluator

client = Client()
openai_client = OpenAI()

compiled_graph = build_graph()

# Reuses the same dataset as eval_single_turn.py — same questions,
# just graded on retrieval quality instead of answer correctness.
dataset = get_or_create_dataset(
    client, name="rag-eval-single-turn", examples=SINGLE_TURN_EXAMPLES, recreate=False
)


def target(inputs: dict) -> dict:
    initial_state = {
        "question": inputs["question"],
        "history": [],
        "session_id": "eval-session",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "score": 0.0,
        "reply": "",
    }
    result = asyncio.run(compiled_graph.ainvoke(initial_state))
    return {
        "answer": result["reply"],
        "reranked_chunks": result["reranked_chunks"],
    }


results = evaluate(
    target,
    data="rag-eval-single-turn",
    evaluators=[retrieval_relevance_evaluator],
    experiment_prefix="rag-retrieval",
)
