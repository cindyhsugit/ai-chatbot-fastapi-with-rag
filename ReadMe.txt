# AI Chatbot with RAG

A FastAPI chatbot that answers questions using **Retrieval-Augmented Generation
(RAG)** — instead of relying only on the LLM's training data, it retrieves
relevant chunks from a custom document, injects them into the prompt, and
generates a grounded answer from that context.

This project extends an earlier plain OpenAI-API chatbot by adding a full RAG
pipeline built from primitives (no LangChain), containerizing it with Docker,
and deploying it live.

## How it works

**Indexing (done once, at startup):**
1. A source document (`the_simpsons_summary.md`) is split into chunks
2. Each chunk is converted into an embedding using OpenAI's
   `text-embedding-3-small` model
3. Embeddings are stored in a FAISS vector index for fast similarity search

**Query time (on every chat message):**
1. The user's question is embedded the same way
2. FAISS returns the top-k most similar chunks
3. Those chunks are inserted into the prompt as context
4. The LLM generates an answer grounded in that retrieved context

## Verifying retrieval actually works

Rather than trusting the output at face value, retrieval correctness was
verified with an adversarial test: a fabricated fact (unrelated to any real
Simpsons trivia) was inserted into the source document, and the chatbot was
asked about it directly. Since the LLM could not have known this fact from
training, a correct answer proves retrieval — not the model's memorized
knowledge — produced the response.

## Tech stack

- **Python / FastAPI** — API and routing
- **OpenAI API** — chat completions and embeddings
- **FAISS** — vector similarity search
- **Docker** — containerization
- **pytest / TestClient** — endpoint testing
- **python-dotenv** — environment variable management

## Project structure

```
main.py                    FastAPI app, routes, chat endpoint
rag_tasks.py                Chunking, embedding, FAISS index, retrieval
the_simpsons_summary.md     Source document used for retrieval
test_main.py                 Endpoint tests
requirements.txt            Python dependencies
Dockerfile                   Container build instructions
templates/                   HTML templates for the chat UI
static/                     CSS and static assets
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create an `apiKey.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

Then start the app:

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000` in your browser.

## Running tests

```bash
pytest
```

## Running with Docker

```bash
docker build -t ai-chatbot-with-rag .
docker run -p 8000:8000 --env-file apiKey.env ai-chatbot-with-rag
```

## What this project demonstrates

- Building a RAG pipeline from first principles (chunking, embeddings, vector
  search, prompt augmentation) rather than only using a high-level framework
- Designing and running a test to verify retrieval is actually influencing
  model output, not just assuming it works
- Containerizing a Python web service and managing secrets safely via
  environment variables rather than hardcoding them
- Debugging real cross-environment issues: Python version mismatches breaking
  dependency installation inside Docker, and Windows-specific file encoding
  errors when reading source documents

## Notes / next steps

- The `/chat` endpoint currently runs synchronously; converting the OpenAI
  calls to `async`/`await` is a planned improvement for handling concurrent
  requests more efficiently
- The FAISS index is currently rebuilt from scratch on every app startup;
  persisting it to disk would avoid redundant embedding calls
