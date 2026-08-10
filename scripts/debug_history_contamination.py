import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(project_root, "apiKey.env"))


import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from app.providers import openai_provider


async def main():
    print("=== Test 1: no prior fabricated fact ===")
    reply1 = await openai_provider.generate_answer_openai(
        prompt="who is Homer Simpson's son",
        history=[],
    )
    print(reply1)

    print("\n=== Test 2: with fabricated-fact history ===")
    reply2 = await openai_provider.generate_answer_openai(
        prompt="who is his son",
        history=[
            HumanMessage(content="What is Homer Simpson's favorite food"),
            AIMessage(
                content="In this universe, Homer Simpson's favorite food is broccoli casserole, not donuts."
            ),
        ],
    )
    print(reply2)
    print("\n=== Test 3: real CONTEXT_ONLY_RULE prompt, empty context ===")
    prompt3 = """
Answer this question using only your own trained knowledge.
If you are not confident, or don't actually know the answer,
respond with exactly: NO_KNOWLEDGE

Context:


Question: who is his son
"""
    reply3 = await openai_provider.generate_answer_openai(
        prompt=prompt3,
        history=[
            HumanMessage(content="What is Homer Simpson's favorite food"),
            AIMessage(
                content="In this universe, Homer Simpson's favorite food is broccoli casserole, not donuts."
            ),
        ],
    )
    print(reply3)

    history_for_test = [
        HumanMessage(content="What is Homer Simpson's favorite food"),
        AIMessage(
            content="In this universe, Homer Simpson's favorite food is broccoli casserole, not donuts."
        ),
    ]

    for i in range(5):
        reply = await openai_provider.generate_answer_openai(
            prompt=prompt3, history=history_for_test
        )
        print(f"Run {i}: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
