# Standard library
# reads environment variables
import os
import time
import logging

# Third-party: FastAPI
from fastapi import FastAPI, HTTPException, Request

# some route will return HTML, not JSON
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# serving files like CSS, JavaScript, and images from a static folder
from fastapi.staticfiles import StaticFiles

# define the shape of data coming into and going out of your API
from pydantic import BaseModel

# Third-party: Langgraph / LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

# Local modules
# loads secrets from a .env file
from dotenv import load_dotenv
from logging_config import setup_logging

# helps build file paths safely.
from pathlib import Path
import providers.openai_provider, providers.gemini_provider
import web_search_provider
import prompt_rules
import graph_builder
import rag_tasks
import utility

# setup
load_dotenv("apiKey.env")
load_dotenv(".env")

import os

# set up logs
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# creates the actual web app
app = FastAPI()

checkpointer = MemorySaver()
graph = graph_builder.build_graph(checkpointer=checkpointer)

# This finds the folder where main.py lives
BASE_DIR = Path(__file__).resolve().parent

# When the browser asks for /static/..., serve files from the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
#  tells FastAPI where the HTML template files are stored. FastAPI’s docs show Jinja2Templates being used exactly for this purpose
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# deprecatd with /chat
# store chat history, list of dictionaries
# history = []
# session_store: dict[str, list] = {}


# data models
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    reply: str


# use langgraph built in fallbacks methods
backup_llm = providers.openai_provider.openai_llm.with_fallbacks(
    [providers.gemini_provider.gemini_llm]
)


# home page route
@app.get("/", response_class=HTMLResponse)
# when visits /, run this function and return HTML. The request is needed for template rendering
def home(request: Request):
    #  loads index.html and sends it to the browser
    return templates.TemplateResponse(
        request=request, name="index_langgraph.html", context={}
    )


@app.get("/langgraph", response_class=HTMLResponse)
async def langgraph(request: Request):
    return templates.TemplateResponse(
        request=request, name="index_langgraph.html", context={}
    )


# OpenAI first, falls back to Gemini on failure
async def generate_with_llm_failover(
    prompt: str,  # system-level rules/instructions for this turn
    messages: list = None,  # conversation history + current question (no rules baked in)
) -> str:

    full_messages = [SystemMessage(content=prompt)] + messages

    response = await backup_llm.ainvoke(full_messages)

    # open ai and gemini returns different response structure
    return utility.unify_response_content.to_text(response.content)


# a tiny check route. If it returns {"status": "ok"}, the server is alive.
@app.get("/health")
def healthAPIEndpoint():
    return {"status": "ok"}


@app.post("/langgraphchat")
async def langgraphchat(request: ChatRequest):
    session_id = request.session_id
    # this becomes the thread_id for the checkpointer later

    # this is the starting state dict of the graph.
    initial_state = {
        "question": request.message,
        "history": [],
        "session_id": session_id,
        "retrieved_chunks": [],
        "score": 0.0,
        "reply": "",
    }

    start = time.time()
    # if any node inside the graph raises an error
    try:
        result = await graph.ainvoke(
            initial_state,
            #
            # config = {
            #     "configurable": {
            #         "thread_id": session_id,      # used by the checkpointer
            #         "user_id": "abc",             # could be used by your own nodes
            #         "some_other_setting": "..."    # anything else your graph reads via configurable
            #     },
            #     "recursion_limit": 50,             # unrelated to checkpointer entirely
            #     "callbacks": [...],                # tracing/logging, also unrelated to checkpointer
            #     "tags": ["prod"],
            # }
            #
            config={"configurable": {"thread_id": session_id}},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.time() - start
    print(f"LangGraph total time: {elapsed:.2f}s")

    return {"reply": result["reply"]}


# run main.py directly, start the server on your computer at port 8000
if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )  # Cloud Run sets PORT=8080; falls back to 8000 for local runs
    uvicorn.run(app, host="0.0.0.0", port=port)
