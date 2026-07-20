
import rag_tasks

test_queries = [
    "What is Homer Simpson's favorite food?",   # known-good
    "What is the secret ingredient in broccoli casserole?",  # known-bad
    "What is the capital of France?",
    "Hello, how are you?",
    "Who is side show bob in the simpsons?",
    "What is the comedy style of the simsons",
    "What is the comedy style of the simpsons"
    
]

#"What is Homer Simpson's favorite food?" -> 
# top score: 0.4053047001361847

# 'What is the secret ingredient in broccoli casserole?' -> 
# top score: -8.455362319946289

# 'What is the capital of France?' -> 
# top score: -10.978986740112305

# 'Hello, how are you?' -> 
# top score: -9.633153915405273

# 'Who is side show bob in the simpsons?' -> 
# top score: -2.488299608230591

# 'What is the comedy style of the simsons' -> 
# top score: -5.348282337188721

# 'What is the comedy style of the simpsons' -> 
# top score: 5.589724063873291

for q in test_queries:
    result = rag_tasks.retrieve(q)
    top_score = result[0][1] if result else None
    print(f"{q!r} -> top score: {top_score}")