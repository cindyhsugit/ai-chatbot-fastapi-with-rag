from sentence_transformers import CrossEncoder

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import time

#  Calculates optimal max_length dynamically for a variable-sized semantic chunk.

#fmt:off
def calculate_max_length(
    chunk_text: str, 
    tokenizer, 
    query_tokens: int = 20, 
    safety_margin: float = 1.15
) -> int:
#fmt:on
    chunk_tokens = len(tokenizer.encode(chunk_text))
    raw_max = chunk_tokens + query_tokens
    return int(raw_max * safety_margin)


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


# Warm-up: absorb the cold-start cost here instead of on a user's first request
_ = onnx_model(**tokenizer("warmup query", "warmup passage", return_tensors="pt"))

# Loaded once at startup
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Takes the user query and a list of candidate chunk texts (from FAISS retrieval),
# returns the top_k chunks reordered by relevance.
def rerank(
    query: str, candidates: list[str], top_k: int = 5
) -> list[tuple[str, float]]:

    # query -> "what's homer's favorite food" (str)
    # candidates -> ["chunk 1 text...", "chunk 2 text...", ...] (list[str])
    #
    # pairs -> [["what's homer's favorite food", "chunk 1 text..."],
    #           ["what's homer's favorite food", "chunk 2 text..."], ...]
    # each pair feeds the cross-encoder query+doc together

    pairs = [[query, candidate] for candidate in candidates]
    scores = model.predict(pairs)

    # Sort candidates by score, descending
    scored = list(zip(candidates, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    # scored looks like
    # [
    #   ("chunk A", 0.95),
    #   ("chunk B", 0.90),
    #   ("chunk C", 0.80)
    # ]

    # result = []
    # for text, score in scored[:top_k]:
    #     result.append(text)
    # result would look like ["chunk A", "chunk B"]
    return scored[:top_k]

#fmt:off
def rerank_with_onnx(
    query: str, 
    retrieved_chunks: list[str], 
    top_k: int = 3
) -> list[tuple[str, float]]:
#fmt:on
       start = time.time()

       # sort the arriving chunk by length first
       sorted_chunks = sorted(retrieved_chunks, key=lambda c: len(c))

        # batch size 5 is 0.01s faster than size 2
       batch_size = 5
       all_scored = []

       for i in range(0, len(sorted_chunks), batch_size):
              batch_chunks = sorted_chunks[i : i + batch_size]
              pairs = [[query, candidate] for candidate in batch_chunks]
       
              batch_max_tokens = max(
                     calculate_max_length(candidate, tokenizer)
                     for candidate in batch_chunks
              )

    #fmt:off
    # Tokenize all pairs together efficiently
              inputs = tokenizer(
                    pairs, 
                     padding=True, 
                     truncation=True, 
                     max_length=batch_max_tokens, # defaults to the model's absolute maximum architectural limit (usually 512 tokens), forcing the ONNX runtime to compute heavy, wasteful self-attention matrices across empty padding space. Capping it at MAX_LENGTH keeps your input tensor shapes tight (e.g., [10, 113]) and keeps CPU inference lightning fast.
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

              batch_scored = list(zip(batch_chunks, [float(s) for s in score_list]))
              all_scored.extend(batch_scored)
    # Pair with candidates and sort descending
       all_scored.sort(key=lambda x: x[1], reverse=True)

       end = time.time()
       print(f"-- -- Hugging face cross encoder with onnx Time: {end-start:.2f}s")

       return all_scored[:top_k]
