import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("apikey.env")

openai_llm = ChatOpenAI(
    model="gpt-4o",  # for failover testing "gpt-1.4o"
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,  # tells the model: always pick the most probable next word, every time
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# # fmt: off
# async def generate_answer_openai(
#     prompt: str, 
#     history: list | None = None, 
#     max_retries: int = 3
# ) -> str:
# # fmt: on
#     print(f"-- generate_answer_openai****")

#     messages = (history or [])  + [HumanMessage(content=prompt)]
#     # LangChain's .with_retry() handles automatically
#     response = await openai_llm.ainvoke(messages)
#     print(f"-- generate_answer_openai response: {response}****")

#     return response.content
