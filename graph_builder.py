# Standard library
from typing import TypedDict, Annotated

# Third-party
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

# Local modules
import rag_tasks
import prompt_rules
import web_search_provider
from reranker_hf import rerank

# Setup
load_dotenv()

# relevant threshold to be considered to use context
RERANK_SCORE_THRESHOLD = 0.0


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


# open AI expect "role" of "user" or "assistant" in message
# input
# history = [
#     HumanMessage(content="What's the capital of France?"),
#     AIMessage(content="The capital of France is Paris."),
#     HumanMessage(content="What's its population?"),
#     AIMessage(content="Paris has a population of about 2.1 million people."),
# ]
# output
# [
#     {"role": "user", "content": "What's the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris."},
#     {"role": "user", "content": "What's its population?"},
#     {"role": "assistant", "content": "Paris has a population of about 2.1 million people."},
# ]


def retrieve_node(state: ChatState) -> dict:

    question = state["question"]
    retrieved_chunks = rag_tasks.retrieve(question, 10)

    return {"retrieved_chunks": retrieved_chunks}


def rerank_node(state: ChatState) -> dict:
    question = state["question"]
    retrieved_chunks = state["retrieved_chunks"]
    reranked = rag_tasks.rerank_with_onnx(question, retrieved_chunks)

    # top score drives the threshold decision in the next node
    top_score = reranked[0][1] if reranked else 0.0

    return {
        "reranked_chunks": reranked,  # list of (text, score) tuples
        "score": top_score,
    }


# keep routing logic lives in its own small separate function
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


# # 2nd node after retrieval node
# async def generate_with_context_node(state: ChatState) -> dict:
#     question = state["question"]
#     chunks = state["reranked_chunks"]  # list of (text, score) tuples
#     history = state["history"]
#     context_text = "\n\n".join(chunk_text for chunk_text, _ in chunks)

#     from main import construct_prompt

#     # fmt:off
#     prompt = construct_prompt(
#         rules=prompt_rules.CONTEXT_ONLY_RULE, 
#         context=context_text, 
#         question=question
#     )
#     # fmt:on#     messages = history + [HumanMessage(content=prompt)]#     from main import generate_with_llm_failover#     reply = await generate_with_llm_failover(prompt=prompt, messages_override=messages)
#     # return only the fields they actually changed
#     return {
#         "reply": reply,
#         "history": [
#             HumanMessage(content=question),
#             AIMessage(content=reply),
#         ],
#     }

async def generate_with_context_node(state: ChatState) -> dict:
    question = state["question"]
    chunks = state["reranked_chunks"]  # list of (text, score) tuples
    history = state["history"]
    context_text = "\n\n".join(chunk_text for chunk_text, _ in chunks)

    # 1. Structure as clear system rules + context + user input
    prompt=f"{prompt_rules.CONTEXT_ONLY_RULE}\n\nContext:\n{context_text}"

    # 2. Construct clean message history with system instructions at the root
    messages = history + [HumanMessage(content=question)]

    from main import generate_with_llm_failover

    # 3. Pass the full message array directly to your failover function
    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)

    return {
        "reply": reply,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }

# 3rd node after retrieval node
async def generate_without_context_node(state: ChatState) -> dict:
    question = state["question"]
    history = state["history"]

    # System rules pinned once at the root, not repeated inline every turn
    prompt = prompt_rules.CONTEXT_TRAINED_DATA_ONLY_RULE

    messages=history + [HumanMessage(content=question)]

    from main import generate_with_llm_failover

    reply = await generate_with_llm_failover(prompt=prompt, messages=messages)

    return {
        "reply": reply,
        "history": [
            HumanMessage(content=question),
            AIMessage(content=reply),
        ],
    }


# 4th node after generation node
async def web_search_node(state: ChatState) -> dict:
    question = state["question"]
    web_results = await web_search_provider.web_search_fallback(question)
    if not web_results:
        return {
            "reply": "I don't know - no local context, no trained knowledge, and web search returned nothing."
        }

    history = state["history"]
    prompt=f"{prompt_rules.WEB_SEARCH_RULE}\n\nWeb results:\n{web_results}"
    messages = history + [HumanMessage(content=question)]

    from main import generate_with_llm_failover

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

    graph.add_node("retrieve_node", retrieve_node)

    graph.add_node("rerank_node", rerank_node)
    graph.add_edge("retrieve_node", "rerank_node")
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.add_node("web_search_node", web_search_node)

    graph.set_entry_point("retrieve_node")
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
