import time
import asyncio
from gemini_provider import generate_answer_gemini  # adjust if your file/function names differ

async def test():
    start = time.time()
    result = await generate_answer_gemini("Say hello in one sentence.")
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(result)

asyncio.run(test())