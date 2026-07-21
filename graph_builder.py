import rag_tasks
from reranker_hf import rerank  

from typing import TypedDict, Annotated, List, Tuple
import operator
import prompt_rules 
from langgraph.graph import StateGraph, END


from openai import OpenAI
from openai import AsyncOpenAI
# reads environment variables
import os

from dotenv import load_dotenv
load_dotenv()

RERANK_SCORE_THRESHOLD = 0.0 

class ChatState(TypedDict):
    question: str
    history: list
    session_id: str
    retrieved_chunks: list
    score: float
    reply: str
    retry_count: int

    
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
        "retrieved_chunks": reranked,   # list of (text, score) tuples
        "score": top_score,
    }

def score_threshold_router(state: ChatState) -> str:
    if state["score"] > 0:
        return "generate_with_context"
    else:
        return "generate_without_context"

   

async def generate_with_context_node(state: ChatState) -> dict:
    question = state["question"]
    chunks = state["retrieved_chunks"]  # list of (text, score) tuples
    history = state["history"] 
    context_text = "\n\n".join(chunk_text for chunk_text, _ in chunks)
    from main import construct_prompt
    prompt = construct_prompt(rules=prompt_rules.CONTEXT_ONLY_RULE, 
                                   context=context_text, 
                                   question=question)


    messages = history + [{"role": "user", "content": prompt}]
    from main import generate_with_network_failover
    reply = await generate_with_network_failover(prompt=prompt, 
                                                 messages_override=messages)

    return {"reply": reply}
   
def generate_without_context_node(state: ChatState) -> dict:
    pass

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)

    graph.set_entry_point("retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        score_threshold_router,
        {
            "generate_with_context": "generate_with_context",
            "generate_without_context": "generate_without_context",
        },
        )
    graph.add_edge("retrieve_and_rerank", END)

    return graph.compile()

graph = build_graph()

# ASCII in the terminal, zero dependencies
print(graph.get_graph().draw_ascii())
# saves a PNG you can open directly
graph.get_graph().draw_mermaid_png(output_file_path="graph_diagram.png")