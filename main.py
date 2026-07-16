from fastapi import FastAPI, HTTPException, Request

# some route will return HTML, not JSON
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates

# serving files like CSS, JavaScript, and images from a static folder
from fastapi.staticfiles import StaticFiles

# define the shape of data coming into and going out of your API
from pydantic import BaseModel

# only the exact values are allowed
from typing import Literal

# reads environment variables
import os

#  client for the OpenAI API
from openai import OpenAI
from openai import AsyncOpenAI

# loads secrets from a .env file
from dotenv import load_dotenv

# helps build file paths safely.
from pathlib import Path

import logging
from logging_config import setup_logging
import gemini_provider

import time

from web_search_provider import web_search_fallback

from prompt_rules import CONTEXT_ONLY_RULE, CONTEXT_TRAINED_DATA_ONLY_RULE

# setup
load_dotenv("apiKey.env")
load_dotenv(".env")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# OpenAI async client
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 

# set up logs
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# creates the actual web app
app = FastAPI()

# This finds the folder where main.py lives
BASE_DIR = Path(__file__).resolve().parent

# When the browser asks for /static/..., serve files from the static folder
app.mount("/static", StaticFiles(directory="static"), name = "static")
#  tells FastAPI where your HTML template files are stored. FastAPI’s docs show Jinja2Templates being used exactly for this purpose
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


#store chat history, list of dictionaries
history = []

# data models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# home page route
@app.get("/" , response_class=HTMLResponse)
# when visits /, run this function and return HTML. The request is needed for template rendering
def home(request: Request):
    #  loads index.html and sends it to the browser
    return templates.TemplateResponse(request = request, 
                                      name = "index.html", 
                                      context = {})


import rag_tasks

async def generate_with_network_failover (prompt: str, messages_override: list = None) -> str:
    """
    Tries OpenAI first, falls back to Gemini on failure.
    Returns the reply string. Used for both the initial answer attempt
    and the grounded (post-web-search) re-generation.
    """
    try:
        start = time.time()
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_override or [{"role": "user", "content": prompt}]
        )
        elapsed = time.time() - start
        print(f"OpenAI response time: {elapsed:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI failed: {e}, falling back to Gemini")
        start = time.time()
        reply = await gemini_provider.generate_answer_gemini(prompt)
        elapsed = time.time() - start
        print(f"Gemini response time: {elapsed:.2f} seconds")
        return reply


async def generate_with_knowledge_failover(question: str, prompt: str) -> str:
    """
    Runs the prompt through generate_with_failover, and if the model
    signals NO_KNOWLEDGE, falls back to web search + a grounded
    regeneration (also via generate_with_failover, so failover applies
    to that call too).
    """
    messages_to_send = history + [{"role": "user", "content": prompt}]
    
    reply = await generate_with_network_failover (prompt, messages_to_send)
    if reply.strip() == "NO_KNOWLEDGE":
        web_results = await web_search_fallback(question)

        if not web_results:
            return "I don't know - no local context, no trained knowledge, and web search returned nothing."

        grounded_prompt = construct_prompt(CONTEXT_ONLY_RULE, question, web_results)
        
        reply = await generate_with_network_failover (grounded_prompt)
        reply = f"{reply}\n\n(Note: answer sourced from live web search, not local knowledge base.)"
    
    return reply


def construct_prompt(rules: str, context: str, question: str) -> str:
    return f"""
{rules}

Context:
{context}

Question: {question}
"""


# use retrieval
@app.post("/chat")
async def chat(request: ChatRequest):

    # grab the pages most related to what the user asked
    # relevant_chunks : list[str]
    start = time.time()
    relevant_chunks = rag_tasks.retrieve(request.message)
    end = time.time()
    print(f"Retrieval Process Till Augmentation Time taken: {end - start:.2f} seconds")
   
    # staples them all into one single block of text, 
    # with a blank line between each card (\n\n means "new line, new line)
    context = "\n\n".join(relevant_chunks)

    augmented_message = construct_prompt(CONTEXT_TRAINED_DATA_ONLY_RULE, request.message, context)
   
    print(f"Prompt length: {len(augmented_message)} characters")
    # Send prior conversation history + this turn's augmented question
    messages_to_send = history + [{"role": "user", "content": augmented_message}]

    # debugging purpose
    #print(json.dumps(messages_to_send, indent=2))  
   
    # response = await async_client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=messages_to_send
    # )
    # reply = response.choices[0].message.content
    
    total_start = time.time()
    reply = await generate_with_knowledge_failover(request.message, augmented_message)
    total_elapsed = time.time() - total_start
    print(f"Total answer_with_coverage_check time: {total_elapsed:.2f} seconds")
    # Save the CLEAN question (not the augmented version) and the reply
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": reply})
  
    return {"reply": reply}


# a tiny check route. If it returns {"status": "ok"}, the server is alive.
@app.get("/health")
def healthAPIEndpoint():
    return {"status": "ok"}

# run main.py directly, start the server on your computer at port 8000
if __name__ == "__main__":
    
    # runs the web server
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
