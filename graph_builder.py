from typing import TypedDict, Optional

class ChatState(TypedDict):
    question: str
    history: list
    session_id: str
    retrieved_chunks: list
    score: float
    reply: str
    retry_count: int