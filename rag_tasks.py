# Standard library
import os
import time
from pathlib import Path

# Third-party
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local modules
import embeddings_hf
import config
import vectorstore_chroma

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch

# Setup
load_dotenv()
load_dotenv("apiKey.env")

# Load ONNX model at startup
model_dir = "./ms-marco-onnx"

options = ort.SessionOptions()
options.intra_op_num_threads = 2  # match Cloud Run's 2 vCPU
options.inter_op_num_threads = 1  # single-model pipeline, no parallel subgraphs
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_model = ORTModelForSequenceClassification.from_pretrained(
    model_dir, provider="CPUExecutionProvider", session_options=options
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


#  calculate max_length for cross-encoder tokenization 512 is too big for 500 chunk size
def calculate_max_length(
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
    avg_query_tokens: int = 20,
    safety_margin: float = 1.15,
) -> int:
    # Use a realistic text sample to measure characters-per-token for this specific tokenizer
    sample_text = "The quick brown fox jumps over the lazy dog. " * 15
    sample_tokens = len(tokenizer.encode(sample_text))
    chars_per_token = len(sample_text) / sample_tokens

    # Max possible chunk size including overlap
    max_chunk_chars = chunk_size + chunk_overlap
    max_chunk_tokens = max_chunk_chars / chars_per_token
    # Add query tokens and apply safety margin
    raw_max = max_chunk_tokens + avg_query_tokens
    return int(raw_max * safety_margin)


MAX_LENGTH = calculate_max_length(config.CHUNK_SIZE, config.CHUNK_OVERLAP, tokenizer)
print(
    f"Derived MAX_LENGTH={MAX_LENGTH} from chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}"
)

# Warm-up: absorb the cold-start cost here instead of on a user's first request
_ = onnx_model(**tokenizer("warmup query", "warmup passage", return_tensors="pt"))


# get_embedding(text: str) -> List[float]:
def get_embedding(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return_list = embeddings_hf.embed_texts([text])[0].tolist()

    return return_list


# Retrieve relevant chunks for a new question
# def retrieve(question: str, k: int = 20) -> list[tuple[str, float]]:
def retrieve(question, k=20):
    # question_embedding -> [0.0119, -0.0440, 0.0801, ...]   (1536 floats, similar to chunk 0's embedding)
    question_embedding = get_embedding(question)

    start = time.time()

    retrieved_chunks = vectorstore_chroma.search(
        query_embedding=question_embedding, k=k
    )

    end = time.time()
    print(f"-- Chroma DB search Time: {end-start:.2f}s")

    return retrieved_chunks


# def rerank(
#     question: str, retrieved_chunks: list[str], top_k: int = 3
# ) -> list[tuple[str, float]]:
#     start = time.time()
#     reranked_chunks = reranker_hf.rerank(
#         question, retrieved_chunks, top_k=3
#     )  # back down to 3 for generation
#     end = time.time()
#     print(f"-- -- Hugging face cross encoder Time: {end-start:.2f}s")

#     # now use reranked_chunks (not retrieved_chunks) when building the prompt for generation
#     return reranked_chunks


def rerank_with_onnx(
    query: str, retrieved_chunks: list[str], top_k: int = 3
) -> list[tuple[str, float]]:

    start = time.time()

    # Build pairs [query, candidate]
    pairs = [[query, candidate] for candidate in retrieved_chunks]

    #fmt:off
    # Tokenize all pairs together efficiently
    inputs = tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH, # defaults to the model's absolute maximum architectural limit (usually 512 tokens), forcing the ONNX runtime to compute heavy, wasteful self-attention matrices across empty padding space. Capping it at MAX_LENGTH keeps your input tensor shapes tight (e.g., [10, 113]) and keeps CPU inference lightning fast.
        return_tensors="pt"
    )
    #fmt:on

    # Run inference with the ONNX model
    with torch.no_grad():
        outputs = onnx_model(**inputs)
        scores = outputs.logits.squeeze(-1)  # Extract raw score logits

    # Convert scores to a Python list
    score_list = (
        scores.tolist() if isinstance(scores, torch.Tensor) else [scores.item()]
    )

    # Pair with candidates and sort descending
    scored = list(zip(retrieved_chunks, [float(s) for s in score_list]))
    scored.sort(key=lambda x: x[1], reverse=True)
    end = time.time()
    print(f"-- -- Hugging face cross encoder with onnx Time: {end-start:.2f}s")

    return scored[:top_k]


# Returns the file's text content, or exits cleanly with a clear message if anything goes wrong
def safely_open_input_file() -> str:
    filename = os.environ.get("INPUT_FILE")
    if not filename:
        raise SystemExit("INPUT_FILE environment variable is not set")

    path = Path(filename)
    if not path.exists():
        raise SystemExit(f"Input file not found: {filename}")

    if path.is_dir():
        raise SystemExit(f"Expected a file, got a directory: {filename}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"Could not decode {filename} as UTF-8: {e}")
    except PermissionError:
        raise SystemExit(f"Permission denied reading file: {filename}")


if __name__ == "__main__":
    print(
        "This file builds a search index when imported — run main.py instead of this file directly."
    )
else:
    loaded_text = safely_open_input_file()

    # initializing chunking style
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_SIZE,
    )

    start = time.time()
    # Generate embeddings for each chunk
    # chunks: List[str] = chunk_text(text)
    # example: ["apple", "banana"]
    # chunks = chunk_text(text)
    chunks = text_splitter.split_text(loaded_text)
    end = time.time()
    print(f"chunk text Time: {end-start:.2f}s")

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
    # Shape: 2 chunks x 384 numbers each for hugging face
    # embeddings: List[List[float]] = [get_embedding(chunk) for chunk in chunks]
    embeddings = []
    start = time.time()
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    end = time.time()

    print(f"Embedding dimension: {len(embeddings[0])}")  # should print 384

    print(f"hugging face embedding Time: {end-start:.2f}s")

    chunk_ids = []
    for i in range(len(chunks)):
        chunk_ids.append(str(i))

    # list comprehension way
    # chunk_ids = [str(i) for i in range(len(chunks))]
    vectorstore_chroma.add_documents(
        ids=chunk_ids,
        embeddings=embeddings,  # this is your HuggingFace embeddings list
        documents=chunks,  # this is your list of chunk text strings
    )
    print("Using ChromaDB:", vectorstore_chroma.__name__)

    print("ChromaDB Collection Count:", vectorstore_chroma.collection.count())
