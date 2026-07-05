from dotenv import load_dotenv
load_dotenv("apiKey.env")

import os
from pathlib import Path
from openai import OpenAI
from openai import AsyncOpenAI
import faiss
import numpy as np

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# using open AI's embedding model
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# different flavor with async 
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async def get_embedding_async(text):
    response = await async_client.embeddings.create(model="text-embedding-3-small", 
                                                    input=text
    )
    return response.data[0].embedding
                                                    

# Retrieve relevant chunks for a new question
async def retrieve_async(question, k=3):
    question_embedding = await get_embedding_async(question)
    distances, indices = index.search(
        np.array([question_embedding]).astype("float32"), k
    )
    return [chunks[i] for i in indices[0]]

if __name__ == "__main__":
     print("This file builds a search index when imported — run main.py instead of this file directly.")
else:
    filename = os.environ["INPUT_FILE"]
    text = Path(filename).read_text(encoding="utf-8")
   # Generate embeddings for each chunk
    chunks = chunk_text(text)
    # Turn each chunk into an embedding (a list of numbers representing
    # its meaning)
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Build the vector index (do this once, at startup or as a script)
    dimension = len(embeddings[0])
    # Create an empty FAISS "filing cabinet" sized to hold embeddings
    # of this exact length. "L2" means it measures similarity using straight-line
    # distance between two points. "Flat" means it checks every stored item
    # directly (simple and accurate, fine for small datasets)
    index = faiss.IndexFlatL2(dimension)

    # Convert our list of embeddings into a NumPy array (a faster format
    # for doing math on lots of numbers), and make sure the numbers are in the
    # exact number format ("float32") that FAISS requires
    # Then, file all of them into the index so they're ready to be searched
    index.add(np.array(embeddings).astype("float32"))
