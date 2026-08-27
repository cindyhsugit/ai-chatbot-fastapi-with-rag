# Standard library
import os
import time
import logging

# Third-party: FastAPI
from fastapi import Form, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Third-party: Langgraph / LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

# Local modules
from dotenv import load_dotenv
from app.logging_config import setup_logging
from pathlib import Path
from app.providers import openai_provider, gemini_provider
from app.text_rag import graph_builder

from app.utility import chroma_sync
from app.utility import rate_limit
from app.utility import access_gate
from app.utility import file_io, chunks_utils, unify_response_content

# check for stale data and clean if needed
chroma_sync.verify_and_clean_chroma()

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

# This finds the folder where main.py lives
BASE_DIR = Path(__file__).resolve().parent

# Read debug mode from environment (defaults to False if not set)
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# When the browser asks for /static/..., serve files from the static folder
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
#  tells FastAPI where the HTML template files are stored. FastAPI’s docs show Jinja2Templates being used exactly for this purpose
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.state.limiter = rate_limit.limiter


# data models
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    reply: str


if DEBUG_MODE:
    print("🛠️ DEBUG_MODE is ON: Routing all traffic to Gemini (Saving OpenAI costs).")
    # Use Gemini exclusively during local debugging
    backup_llm = gemini_provider.gemini_llm
else:
    # Production / Normal mode: OpenAI primary with Gemini fallback
    backup_llm = openai_provider.openai_llm.with_fallbacks([gemini_provider.gemini_llm])


@app.get("/demo-login")
async def demo_login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="demo_login.html", context={"error": None}
    )


@app.post("/demo-login")
async def demo_login_submit(
    request: Request,
    name: str = Form(...),
    company: str = Form(...),
    keyword: str = Form(...),
):
    if not access_gate.check_keyword(keyword):
        return templates.TemplateResponse(
            request=request,
            name="demo_login.html",
            context={
                "error": "Incorrect keyword. Check the demo instructions and try again."
            },
        )
    access_gate.log_visitor(name, company, request)
    response = RedirectResponse(url="/langgraph", status_code=303)
    response.set_cookie("demo_access", "granted", max_age=60 * 60 * 2)
    return response


# home page route
@app.get("/", response_class=HTMLResponse)
# when visits /, run this function and return HTML. The request is needed for template rendering
def home(request: Request):
    #  loads index.html and sends it to the browser
    return templates.TemplateResponse(
        request=request, name="demo_login.html", context={}
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
    return unify_response_content.to_text(response.content)


# a tiny check route. If it returns {"status": "ok"}, the server is alive.
@app.get("/health")
def healthAPIEndpoint():
    return {"status": "ok"}


@app.post("/langgraphchat")
@rate_limit.limiter.limit("5/minute")
async def langgraphchat(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(...),
):

    # this becomes the thread_id for the checkpointer later

    rate_limit.enforce_session_interval(session_id)

    # this is the starting state dict of the graph.
    initial_state = {
        "question": message,
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


checkpointer = MemorySaver()
graph = graph_builder.build_graph(checkpointer=checkpointer)

# run main.py directly, start the server on your computer at port 8000
if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )  # Cloud Run sets PORT=8080; falls back to 8000 for local runs
    uvicorn.run(app, host="0.0.0.0", port=port)
