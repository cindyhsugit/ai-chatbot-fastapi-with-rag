import os
from google import genai
from dotenv import load_dotenv
import time
import asyncio
import graph_builder

load_dotenv("apikey.env")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

async def generate_answer_gemini(prompt: str, history: list | None = None, max_retries: int = 3) -> str:
    messages = (history or []) + [{"role": "user", "content": prompt}]
    gemini_contents = graph_builder.convert_to_gemini_messages(messages)

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.5-flash",
                contents=gemini_contents
            )
            return response.text
        # exponential backoff
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s - exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise  # give up after max_retries, let the error surface