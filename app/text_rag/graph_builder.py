import os
import time
from pathlib import Path

# Standard library
from typing import TypedDict, Annotated
from collections import defaultdict

# Third-party
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_community.retrievers import BM25Retriever
import ollama
import sqlite3

# Local modules
from app.text_rag import prompt_rules
from app.providers import web_search_provider
from app.providers import ollama_provider
from app.text_rag import reranker_hf
import app.text_rag.embeddings_hf as embeddings_hf
import app.config as config
from app.text_rag import vectorstore_chroma
from app.text_rag import chunking_strategy
from app.utility import file_io
from app.utility import sql_regex_check
from app.utility import reciprocal_rank_fusion

# Setup
load_dotenv()


class ChatState(TypedDict):
    question: str
    history: Annotated[  # Annotated is generic Python typing syntax, as a way to attach their own special instructions onto a type hint
        list, add_messages  # add_messages reducer function merges history
    ]
    session_id: str
    retrieved_chunks: list
    reranked_chunks: list
    score: float  # top cross-encoder reranker score from retrieved_chunks
    reply: str
    query_route: str


# 1 node determine query can be converted to SQL search or regular semantic
def modal_node(state: ChatState) -> dict:
    question = state["question"]

    start = time.time()
    if sql_regex_check.is_sql_intent(question):
        end = time.time()
        # print(f"Fast Regex Determinant Time: {end - start:.4f}s -- is_sql : True")
        return {"query_route": "sql"}

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Can this question be answered by converting it into a SQL query "
                    "against a database of Simpsons characters and episodes (counts, "
                    "filters, exact numbers, comparisons)? Reply with exactly one word: "
                    "yes or no."
                ),
            },
            {"role": "user", "content": question},
        ],
        options={
            "temperature": 0,
            "num_predict": 2,
        },
    )

    answer = response["message"]["content"].strip().lower()
    is_sql = "yes" in answer

    end = time.time()
    # print(f"local LLM modal determin Time: {end-start:.2f}s -- is_sql : {is_sql}")

    return {"query_route": "sql" if is_sql else "semantic"}


# 2 level node
def retrieve_node(state: ChatState) -> dict:

    question = state["question"]

    question_embedding = embeddings_hf.get_embedding(question)

    pool_k = 15

    start = time.time()
    chroma_pool = vectorstore_chroma.search(
        query_embedding=question_embedding, k=pool_k
    )

    # print(f"-- Chroma results ({len(chroma_pool)}):")
    # for r in chroma_pool:
    #     print(f"   {r[:80]}")

    bm25_docs = bm25_retriever.invoke(question)
    bm25_pool = [doc.page_content for doc in bm25_docs][:pool_k]

    # print(f"-- BM25 results ({len(bm25_pool )}):")
    # for r in bm25_pool:
    #     print(f"   {r[:80]}")

    # seen = set()
    # retrieved_chunks = []

    # def fill_from(pool, count_needed):
    #     filled = []
    #     for candidate in pool:
    #         if len(filled) >= count_needed:
    #             break
    #         if candidate not in seen:
    #             seen.add(candidate)
    #             filled.append(candidate)
    #     return filled

    # chroma_chunks = fill_from(chroma_pool, target_k)
    # bm25_chunks = fill_from(bm25_pool, target_k)
    # retrieved_chunks = chroma_chunks + bm25_chunks

    # 3. Apply RRF fusion module
    target_k = 10

    retrieved_chunks = reciprocal_rank_fusion.combine(
        chroma_pool, bm25_pool, rrf_k=60, top_n=target_k
    )

    end = time.time()
    print(
        f"-- RRF retrieved {len(retrieved_chunks)} combined chunks in {end-start:.2f}s"
    )

    return {"retrieved_chunks": retrieved_chunks}


# 2 level SQL Processing Node
async def sql_search_node(state: ChatState) -> dict:
    question = state["question"]

    # 1. Generate SQL via LLM or helper function
    raw_sql = await ollama_provider.generate_sql_with_llm(question)
    print(f"-- raw_sql : {raw_sql}")
    # 2. Execute against database
    # Example placeholder response:
    sql_result_data = f"Executed SQL lookup for: {question}"

    history = state["history"]
    prompt = f"Answer the user query based on this database result:\n{sql_result_data}"
    messages = history + [HumanMessage(content=question)]

    from app.main import generate_with_llm_failover

    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)

    return {
        "reply": reply,
        "sql_results": sql_result_data,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }


# 3rd node
def rerank_node(state: ChatState) -> dict:
    question = state["question"]
    retrieved_chunks = state["retrieved_chunks"]
    reranked = reranker_hf.rerank_with_onnx(question, retrieved_chunks)

    # top score drives the threshold decision in the next node
    top_score = reranked[0][1] if reranked else 0.0

    return {
        "reranked_chunks": reranked,  # list of (text, score) tuples
        "score": top_score,
    }


# keep routing logic lives in its own small separate function
def score_threshold_router(state: ChatState) -> str:
    if state["score"] > config.RERANK_SCORE_THRESHOLD:
        return "generate_with_context"
    else:
        return "generate_without_context"


def not_in_context_router(state: ChatState) -> str:
    if state["reply"] == "NO_KNOWLEDGE":
        return "web_search_node"
    else:
        return "END"


def sql_or_semantic_router(state: ChatState) -> str:
    return state["query_route"]


# 4th node after retrieval node
async def generate_with_context_node(state: ChatState) -> dict:
    question = state["question"]
    chunks = state["reranked_chunks"]  # list of (text, score) tuples
    history = state["history"]
    context_text = "\n\n".join(chunk_text for chunk_text, _ in chunks)

    # 1. Structure as clear system rules + context + user input
    prompt = f"{prompt_rules.CONTEXT_ONLY_RULE}\n\nContext:\n{context_text}"

    # 2. Construct clean message history with system instructions at the root
    messages = history + [HumanMessage(content=question)]

    from app.main import generate_with_llm_failover

    start = time.time()
    # 3. Pass prompt and message over to generate
    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)
    end = time.time()
    print(f"-- generate_with_context_node Time: {end-start:.2f}s")

    return {
        "reply": reply,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }


# 4th node after retrieval node
async def generate_without_context_node(state: ChatState) -> dict:
    question = state["question"]
    history = state["history"]

    # System rules pinned once at the root, not repeated inline every turn
    prompt = prompt_rules.CONTEXT_TRAINED_DATA_ONLY_RULE

    messages = history + [HumanMessage(content=question)]

    from app.main import generate_with_llm_failover

    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)

    return {
        "reply": reply,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }


# 5th node after generation node
async def web_search_node(state: ChatState) -> dict:
    question = state["question"]
    web_results = await web_search_provider.web_search_fallback(question)
    if not web_results:
        return {
            "reply": "I don't know - no local context, no trained knowledge, and web search returned nothing."
        }

    history = state["history"]
    prompt = f"{prompt_rules.WEB_SEARCH_RULE}\n\nWeb results:\n{web_results}"
    messages = history + [HumanMessage(content=question)]

    from app.main import generate_with_llm_failover

    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)
    reply = f"{reply}\n\n(Note: answer sourced from live web search, not local knowledge base.)"

    return {
        "reply": reply,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }


def build_graph(checkpointer=None) -> CompiledStateGraph:
    # Initialize the Graph
    graph = StateGraph(ChatState)

    # Add Nodes
    graph.add_node("modal_node", modal_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("sql_search_node", sql_search_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_edge("retrieve_node", "rerank_node")
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.add_node("web_search_node", web_search_node)

    # Set Entry Point
    graph.set_entry_point("modal_node")

    # Conditional routing from router head node
    graph.add_conditional_edges(
        "modal_node",
        sql_or_semantic_router,
        {
            "sql": "sql_search_node",
            "semantic": "retrieve_node",
        },
    )

    graph.add_conditional_edges(
        "rerank_node",
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
    # SQL Node terminates at END
    graph.add_edge("sql_search_node", END)

    # Compile the graph into a runnable application
    # attach a checkpointer (like MemorySaver) to enable
    # thread persistence, state checkpointing, and
    # conversation memory.
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # debug-only
    from langgraph.checkpoint.memory import MemorySaver

    debug_graph = build_graph(MemorySaver())
    print(debug_graph.get_graph().draw_ascii())
    debug_graph.get_graph().draw_mermaid_png(output_file_path="graph_diagram.png")
else:
    # Loading
    input_dir = Path(os.getenv("INPUT_DIR"))

    all_chunks = []
    for filepath in sorted(input_dir.glob("*.md")):
        loaded_text = file_io.safely_open_input_file(str(filepath))
        chunks = chunking_strategy.get_sem_fs_chunk(loaded_text)

        all_chunks.extend(chunks)

    # Embedding
    embeddings = embeddings_hf.embed_chunks(all_chunks)

    # Indexing
    vectorstore_chroma.add_documents(embeddings, all_chunks)

    # BM25 index — same chunks, no re-chunking
    bm25_retriever = BM25Retriever.from_texts(all_chunks)
