import os
from google import genai
from dotenv import load_dotenv

load_dotenv("apikey.env")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def generate_answer_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3-flash-preview", #gemini-3.5-flash
        contents=prompt
    )
    return response.text