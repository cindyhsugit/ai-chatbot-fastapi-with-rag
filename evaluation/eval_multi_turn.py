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

# Update this import to match your actual filename (the file containing build_graph)
from graph_builder import build_graph
from evaluation.datasets import get_or_create_dataset, MULTI_TURN_EXAMPLES
from evaluation.evaluators import correctness_evaluator, turn_attribution_evaluator

client = Client()
openai_client = OpenAI()

# Build once, reuse across all eval examples — no checkpointer, each question is independent
compiled_graph = build_graph()

# 1. Dataset — from shared datasets.py
#fmt:off
dataset = get_or_create_dataset(
    client, name="rag-eval-multi-turn", 
    examples=MULTI_TURN_EXAMPLES, 
    recreate=True
)
#fmt:on


# 2. Target — history passed through, NOT hardcoded to []
def target(inputs: dict) -> dict:
    initial_state = {
        "question": inputs["question"],
        "history": inputs.get("history", []),
        "session_id": "eval-session",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "score": 0.0,
        "reply": "",
    }
    result = asyncio.run(compiled_graph.ainvoke(initial_state))
    return {"answer": result["reply"]}


# 3. Run — both evaluators, since we care about correctness AND turn attribution
results = evaluate(
    target,
    data="rag-eval-multi-turn",
    evaluators=[correctness_evaluator, turn_attribution_evaluator],
    experiment_prefix="multi-turn",
)
