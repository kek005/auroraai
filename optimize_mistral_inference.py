import torch
import time
import numpy
import scipy
import bitsandbytes as bnb
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0))
print("NumPy version:", numpy.__version__)
print("SciPy version:", scipy.__version__)

# Ensure GPU is available
if not torch.cuda.is_available():
    sys.exit("❌ GPU Required! Exiting...")

# Use a BitsandBytes-compatible model
MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"

print(f"Using device: cuda")

# Define BitsAndBytes 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the model
print("🚀 Loading model with BitsandBytes 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Auto-dispatches to GPU
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

print("✅ Model loaded successfully on GPU")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Benchmark function
def benchmark():
    prompt = "What are the benefits of AI in healthcare?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=100)

    print("Running benchmark...")
    times = []
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=100)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"🔥 Avg Inference Time: {avg_time:.3f} seconds")

# Run benchmark
benchmark()
