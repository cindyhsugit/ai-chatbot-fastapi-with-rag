import os
from google import genai
from dotenv import load_dotenv
import time
import asyncio
import graph_builder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv("apikey.env")

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0,
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)


async def generate_answer_gemini(
    prompt: str, history: list | None = None, max_retries: int = 3
) -> str:
    messages = (history or []) + [HumanMessage(content=prompt)]

    response = await gemini_llm.ainvoke(messages)
    return response.content
