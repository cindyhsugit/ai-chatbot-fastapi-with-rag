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

# loads secrets from a .env file
from dotenv import load_dotenv

# helps build file paths safely.
from pathlib import Path

import logging
from logging_config import setup_logging

# setup
load_dotenv("apiKey.env")

# what does the client includes? what is the OpenAI class
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# chat endpoint
@app.post("/chat", response_model=ChatResponse)
# when sends a message to /chat, run this function. response_model=ChatResponse tells FastAPI the reply should look like {"reply": "..."}.
def chatAPIEndpoint(request: ChatRequest):
    logger.info("chat start")
    # calls an external API, and external calls can fail for lots of reasons, so wrapping that code in try / except keeps the server from crashing and lets you return a clean 500 error instead
    try:
        # takes the old messages, need to summarize or drop history if it gets too long 
        messages = history
        # adds the new user message to the list
        messages.append({"role": "user", 
                         "content": request.message})
        # sends the conversation to the OpenAI model and asks for a reply. The chat-completions endpoint is how you ask the model to continue a message list.
        response = client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages = messages
        )
        # pulls out the actual text answer from the model’s response
        first_choice = response.choices[0].message.content
        if first_choice is not None:
            reply = response.choices[0].message.content
        else:
            reply = "No answer returned."

        # seperate user message and bot message?
        # saves both the user message and the bot reply in memory so future replies can use them
        history.append({"role":"user", 
                        "content": request.message})
        history.append({"role":"assistant", 
                        "content": reply})

        logger.info("chat end")
        # sends the reply back to the browser.
        return {"reply": reply}
    # If something goes wrong, this turns the problem into a clean HTTP 500 error instead of crashing silently. FastAPI uses HTTPException for this kind of controlled failure.
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# a tiny check route. If it returns {"status": "ok"}, the server is alive.
@app.get("/health")
def healthAPIEndpoint():
    return {"status": "ok"}

# run main.py directly, start the server on your computer at port 8000
if __name__ == "__main__":
    
    # runs the web server
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
