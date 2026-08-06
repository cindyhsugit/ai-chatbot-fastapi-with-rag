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
from evaluation.datasets import get_or_create_dataset, SINGLE_TURN_EXAMPLES

client = Client()
openai_client = OpenAI()


def correctness_evaluator(run, example) -> dict:
    output = run.outputs["answer"]
    reference = example.outputs["answer"]
    question = example.inputs["question"]

    prompt = f"""Question: {question}
Reference answer (ground truth): {reference}
Model's answer: {output}

Score the model's answer as correct (1) if it clearly states the same
core fact as the reference answer, even if it is phrased differently,
includes extra context, or explicitly rules out other possibilities.
Score it incorrect (0) only if it fails to state the core fact from
the reference, or states something that actively conflicts with it.


Respond with only "1" or "0"."""

    result = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = result.choices[0].message.content.strip()
    try:
        score = int(raw)
    except ValueError:
        print(f"Unexpected judge output: {raw!r}")
        score = 0
    return {"key": "correctness", "score": score}


def turn_attribution_evaluator(run, example) -> dict:
    """
    LLM-as-judge: checks whether the model answered the CURRENT question,
    not a prior one from the conversation history. This directly tests
    the multi-turn flattened-message bug fix.
    """
    output = run.outputs["answer"]
    current_question = example.inputs["question"]
    history = example.inputs.get("history", [])

    if not history:
        return {"key": "turn_attribution", "score": 1}  # nothing to drift to

    prior_questions = "\n".join(m["content"] for m in history if m["role"] == "user")

    prompt = f"""Current question: {current_question}
Prior questions in this conversation:
{prior_questions}

Model's answer: {output}

Does the model's answer address the CURRENT question specifically,
rather than re-answering only a prior question? If the current question
asks for multiple things, all parts should be addressed. Respond with
only "1" (answered current question) or "0" (answered a prior question
or missed part of the current one)."""

    result = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = result.choices[0].message.content.strip()
    try:
        score = int(raw)
    except ValueError:
        print(f"Unexpected judge output: {raw!r}")
        score = 0

    return {"key": "turn_attribution", "score": score}
