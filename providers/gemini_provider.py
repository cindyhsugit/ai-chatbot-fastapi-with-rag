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
)


async def generate_answer_gemini(
    prompt: str, history: list | None = None, max_retries: int = 3
) -> str:
    messages = (history or []) + [HumanMessage(content=prompt)]

    for attempt in range(max_retries):
        try:
            response = await gemini_llm.ainvoke(messages)
            return response.text
        # exponential backoff
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # 1s, 2s, 4s - exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise  # give up after max_retries, let the error surface
