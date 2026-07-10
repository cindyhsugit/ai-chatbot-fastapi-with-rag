from dotenv import load_dotenv
load_dotenv("apiKey.env")

import os
from pathlib import Path
from openai import OpenAI
from openai import AsyncOpenAI
import faiss
import numpy as np
import embeddings_hf as Embeddings_HF
import reranker_hf as Reranker_HF


# Chunk text
# chunk_text(text: str, chunk_size: int = 500) -> List[str]:
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    # range(start, stop, step) — here start=0, stop=len(words) 
    # (the total number of words), step=chunk_size (500)
    for i in range(0, len(words), chunk_size):
       
        # grab words starting at position i, up to (but not including) position i + chunk_size
        slicedWords = words[i:i + chunk_size]

        chunk = " ".join(slicedWords)
        chunks.append(chunk)
    return chunks



#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# get_embedding(text: str) -> List[float]:
def get_embedding(text):
    # using open AI's embedding model
    # response = client.embeddings.create(
    #     model="text-embedding-3-small",
    #     input=text
    # )
    # return_list = response.data[0].embedding
    # replace with huggingFace embedding module
    return_list = Embeddings_HF.embed_texts([text])[0] 
    print(f"Embedding dimension: {len(return_list)}")  # should print 384
    
    return return_list

# different flavor with async 
# async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# async def get_embedding_async(text):
#    # was: response = await async_client.embeddings.create(model="text-embedding-3-small", input=text)
#     return_list = Embeddings_HF.embed_texts([text])[0]
#     return return_list

                                                 

# Retrieve relevant chunks for a new question
# def retrieve(question: str, k: int = 20) -> list[str]:
def retrieve(question, k=20):
    # question_embedding -> [0.0119, -0.0440, 0.0801, ...]   (1536 floats, similar to chunk 0's embedding)
    question_embedding = get_embedding(question)
    
    # indices -> array([[0, 1, -1]])   <- FAISS returns positions in the index
    #   0 = chunk 0 ("The sky is blue.") was the closest match
    #   1 = chunk 1 was second closest
    #  -1 = "no third result" since we only have 2 chunks stored but asked for k=3
    # distances -> array([[0.0012, 0.1583, some_huge_or_placeholder_number]])
    # how far away that vector was from your query vector
    #  could be useful for debugging or setting a "confidence threshold" — e.g., 
    # "if even the closest match has a huge distance, 
    #  maybe there's no relevant document at all
    distances, indices = index.search(
        np.array([question_embedding]).astype("float32"), k=20) # cast a wider net

    # indices[0] = [5, 12, 3, 8, 19, 0, 45, ...]  # 20 numbers total
    # chunks[0] -> "apple"
    # chunks[1] -> "banana"
    # For each number i inside indices[0], go grab chunks[i] 
    # result -> ["apple", "banana", <IndexError risk on -1!>]

    result = []
    #  the first and only row, just 1 question now...
    for i in indices[0]:
        matching_chunk = chunks[i]
        result.append(matching_chunk)

    #list comprehension style 
    #result = [chunks[i] for i in indices[0]]

    # question -> "what's homer's favorite food" (str)
    # result -> ["chunk about donuts...", "chunk about broccoli...", ... ] (list[str], 20 items)
    
    # top_k=3 -> how many we want back after reranking
    # reranked_chunks -> ["chunk about broccoli...", "chunk about donuts...", "chunk about..."] (list[str], 3 items, reordered by relevance)
    reranked_chunks = Reranker_HF.rerank(question, result, top_k=3) # back down to 3 for generation
    print(f"Before rerank: {len(result)} chunks")
    print(f"After rerank: {len(reranked_chunks)} chunks")
    # now use reranked_chunks (not retrieved_chunks) when building the prompt for generation
    return reranked_chunks

if __name__ == "__main__":
     print("This file builds a search index when imported — run main.py instead of this file directly.")
else:
    filename = os.environ["INPUT_FILE"]
    text = Path(filename).read_text(encoding="utf-8")
   
    # Generate embeddings for each chunk
    # chunks: List[str] = chunk_text(text)
    # example: ["apple", "banana"]
    chunks = chunk_text(text)


    # Turn each chunk into an embedding (a list of numbers representing
    # its meaning)
    # get_embedding("apple")    -> [0.0123, -0.0456, 0.0788, ...]  (1536 numbers)
    # get_embedding("banana")   -> [0.0341, 0.0021, -0.0999, ...] (1536 numbers)
    #
    # So embeddings ends up looking like:
    # embeddings = [
    #     [0.0123, -0.0456, 0.0788, ...],   <- embedding for chunk 0 (1536 floats)
    #     [0.0341,  0.0021, -0.0999, ...],  <- embedding for chunk 1 (1536 floats)
    # ]
    # Shape: 2 chunks x 384 numbers each
    # embeddings: List[List[float]] = [get_embedding(chunk) for chunk in chunks]
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Build the vector index (do this once, at startup or as a script)
    # dimension: int = len(embeddings[0])
    # len(embeddings[0]) = len([0.0123, -0.0456, 0.0788, ...]) = 1536
    # dimension = 384 for huggingface
    dimension = len(embeddings[0])

    # Create an empty FAISS "filing cabinet" sized to hold embeddings
    # of this exact length. "L2" means it measures similarity using straight-line
    # distance between two points. "Flat" means it checks every stored item
    # directly (simple and accurate, fine for small datasets)
    # index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
    index = faiss.IndexFlatL2(dimension)

    # Convert our list of embeddings into a NumPy array (a faster format
    # for doing math on lots of numbers), and make sure the numbers are in the
    # exact number format ("float32") that FAISS requires
    # Then, file all of them into the index so they're ready to be searched
    # return None
    # index: faiss.IndexFlatL2
    #  np.array(embeddings) turns the Python list-of-lists into:
    #   array([[0.0123, -0.0456, 0.0788, ...],
    #          [0.0341,  0.0021, -0.0999, ...]])
    # (FAISS requires float32 specifically)
    index.add(np.array(embeddings).astype("float32"))
