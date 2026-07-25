import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

openai_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,  # tells the model: always pick the most probable next word, every time
)


async def generate_answer_openai(
    prompt: str, history: list | None = None, max_retries: int = 3
) -> str:
    messages = (history or []) + [HumanMessage(content=prompt)]

    for attempt in range(max_retries):
        try:
            response = await openai_llm.ainvoke(messages)
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2**attempt)
