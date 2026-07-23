import rag_tasks
from reranker_hf import rerank

from typing import TypedDict, Annotated, List, Tuple
import operator
import prompt_rules
from langgraph.graph import StateGraph, END
import web_search_provider
from langchain_core.messages import BaseMessage
from openai import OpenAI
from openai import AsyncOpenAI

# reads environment variables
import os

from dotenv import load_dotenv

load_dotenv()

RERANK_SCORE_THRESHOLD = 0.0

from typing import Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    question: str
    history: Annotated[list, add_messages]
    session_id: str
    retrieved_chunks: list
    score: float
    reply: str
    retry_count: int

#open AI expect "role" of "user" or "assistant" in message
#input
# history = [
#     HumanMessage(content="What's the capital of France?"),
#     AIMessage(content="The capital of France is Paris."),
#     HumanMessage(content="What's its population?"),
#     AIMessage(content="Paris has a population of about 2.1 million people."),
# ]
#output
# [
#     {"role": "user", "content": "What's the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris."},
#     {"role": "user", "content": "What's its population?"},
#     {"role": "assistant", "content": "Paris has a population of about 2.1 million people."},
# ]


def convert_to_gemini_messages(history: list[BaseMessage | dict]) -> list[dict]:
    result = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            role = "human" if msg.type == "human" else "ai"
            content = msg.content

        gemini_role = "model" if role in ("assistant", "ai") else "user"
        result.append({"role": gemini_role, "parts": [{"text": content}]})
    return result



def convert_to_openai_messages(history: list[BaseMessage | dict]) -> list[dict[str, str]]:
    result = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
        result.append({"role": role, "content": content})
    return result

def retrieve_and_rerank_node(state: ChatState) -> dict:
    """
    Combined retrieve + rerank node.
    Pulls top-20 candidates from ChromaDB, reranks down to top-3,
    and keeps the reranker's scores so the next node (score threshold)
    can decide whether to use this context at all.
    """
    question = state["question"]

    # existing retrieval call, untouched

    # existing rerank call — assumes you've already applied the fix
    # to return (text, score) tuples instead of text-only
    reranked = rag_tasks.retrieve(question)

    # top score drives the threshold decision in the next node
    top_score = reranked[0][1] if reranked else 0.0

    return {
        "retrieved_chunks": reranked,  # list of (text, score) tuples
        "score": top_score,
    }


def score_threshold_router(state: ChatState) -> str:
    if state["score"] > 0:
        return "generate_with_context"
    else:
        return "generate_without_context"


def not_in_context_router(state: ChatState) -> str:
    if state["reply"] == "NO_KNOWLEDGE":
        return "web_search_node"
    else:
        return "END"


async def generate_with_context_node(state: ChatState) -> dict:
    question = state["question"]
    chunks = state["retrieved_chunks"]  # list of (text, score) tuples
    history = state["history"]
    context_text = "\n\n".join(chunk_text for chunk_text, _ in chunks)

    from main import construct_prompt

    prompt = construct_prompt(
        rules=prompt_rules.CONTEXT_ONLY_RULE, context=context_text, question=question
    )

    messages = history + [{"role": "user", "content": prompt}]

    from main import generate_with_llm_failover
    
    reply = await generate_with_llm_failover(
        prompt=prompt, messages_override=convert_to_openai_messages(messages)
    )

    return {"reply": reply,
             "history": [
                 {"role": "user", "content": question},
                 {"role": "assistant", "content": reply},
                 ],
                 }


async def generate_without_context_node(state: ChatState) -> dict:
    question = state["question"]

    history = state["history"]
    from main import construct_prompt

    prompt = construct_prompt(
        rules=prompt_rules.NO_CONTEXT_RULE, context="", question=question
    )

    messages = history + [{"role": "user", "content": prompt}]
    from main import generate_with_llm_failover

    reply = await generate_with_llm_failover(
        prompt=prompt, messages_override=convert_to_openai_messages(messages)
    )
    return {"reply": reply, 
            "history": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
                ],
                }


async def web_search_node(state: ChatState) -> dict:
    question = state["question"]
    web_results = await web_search_provider.web_search_fallback(question)
    if not web_results:
        return {
            "reply": "I don't know - no local context, no trained knowledge, and web search returned nothing."
        }

    from main import construct_prompt

    prompt = construct_prompt(prompt_rules.WEB_SEARCH_RULE, web_results, question)

    history = state["history"]
    messages = history + [{"role": "user", "content": prompt}]
    from main import generate_with_llm_failover

    reply = await generate_with_llm_failover(
        prompt=prompt, messages_override=convert_to_openai_messages(messages)
    )
    reply = f"{reply}\n\n(Note: answer sourced from live web search, not local knowledge base.)"
    
    return {"reply": reply,
            "history": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
                ],
                }


def build_graph(checkpointer=None):
    graph = StateGraph(ChatState)

    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.add_node("web_search_node", web_search_node)

    graph.set_entry_point("retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
    )
    graph.add_conditional_edges(
        "generate_without_context",
        not_in_context_router,
        {"web_search_node": "web_search_node", "END": END},
    )

    graph.add_edge("web_search_node", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    debug_graph = build_graph(MemorySaver())
    print(debug_graph.get_graph().draw_ascii())
    debug_graph.get_graph().draw_mermaid_png(output_file_path="graph_diagram.png")