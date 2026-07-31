# Step 1: One-time export (do this once, save the result)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
output_dir = "./ms-marco-onnx"

# This loads the PyTorch model AND converts it to ONNX in one step
onnx_model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save it so you don't have to re-export every time you start the app
onnx_model.save_pretrained("./ms-marco-onnx")
tokenizer.save_pretrained("./ms-marco-onnx")

print(f"Successfully exported and saved to {output_dir}")
