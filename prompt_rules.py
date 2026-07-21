"""
Rule text used by construct_prompt() to control how strictly the model
must stick to provided context. Kept separate from main.py so the
orchestration logic isn't buried under large prompt strings.
"""

CONTEXT_ONLY_RULE = """
Answer the question using ONLY the context below.
If the context does not directly answer the question, say so in one sentence —
do not list unrelated details from the context.
"""

CONTEXT_TRAINED_DATA_ONLY_RULE = """
Answer the question using the context below if it contains the answer.

If the context does NOT contain the answer, you may answer using your own
trained knowledge instead — but only if you are confident and not guessing.

If neither the context nor your own reliable knowledge can answer this
question, respond with exactly: NO_KNOWLEDGE
"""

NO_CONTEXT_RULE = """
Answer this question using only your own trained knowledge.
If you are not confident, or don't actually know the answer,
respond with exactly: NO_KNOWLEDGE
"""