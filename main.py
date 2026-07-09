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
import json

# use retrieval
@app.post("/chat")
async def chat(request: ChatRequest):

    # grab the pages most related to what the user asked
    # relevant_chunks : list[str]
    relevant_chunks = rag_tasks.retrieve(request.message)
    
    # staples them all into one single block of text, 
    # with a blank line between each card (\n\n means "new line, new line)
    context = "\n\n".join(relevant_chunks)

    augmented_message = f"""Answer the question using only the context below.
            If the answer isn't in the context, say you don't know.

            Context:
            {context}

            Question: {request.message}
            """

    # Send prior conversation history + this turn's augmented question
    messages_to_send = history + [{"role": "user", "content": augmented_message}]

    # debugging purpose
    #print(json.dumps(messages_to_send, indent=2))  
   
    # response = await async_client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=messages_to_send
    # )
    # reply = response.choices[0].message.content
    # # Save the CLEAN question (not the augmented version) and the reply
    # history.append({"role": "user", "content": request.message})
    # history.append({"role": "assistant", "content": reply})

    reply = gemini_provider.generate_answer_gemini(augmented_message)
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
    
    
