import torch
import time
import sys
import numpy as np
import scipy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ✅ Print System Info
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# 🚀 Ensure GPU is available
if not torch.cuda.is_available():
    sys.exit("❌ GPU Required! Exiting...")

# 🏆 Use an optimized model for vLLM
MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"

# 🔥 Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 🚀 Load Model (vLLM keeps it hot in VRAM)
print("🚀 Loading model into VRAM with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.97,  # 🔥 Use max VRAM
    max_num_seqs=2,  # 🔥 Reduce batch overhead
    enforce_eager=True,  # 🔥 Avoid CUDA graph overhead
)

# ✅ Fixed system message for structured responses
system_prompt = (
    "You are an AI interview assistant. Respond concisely and in a structured format. "
    "If a question requires listing multiple items, provide only a bullet-point list, "
    "without explanations or unnecessary details."
)

# ✅ Optimized sampling parameters for fast inference
sampling_params = SamplingParams(
    max_tokens=50,
    temperature=0.1,  # 🔥 Low = deterministic response
    top_p=0.8,  # 🔥 Ensures balanced output
    best_of=1,
)

# 🔥 Warm-up with 50 dummy requests
print("🔥 Warming up the model...")
for i in range(50):
    start_time = time.time()
    llm.generate(system_prompt + "\n\nWarm-up question", sampling_params)
    elapsed = time.time() - start_time
    print(f"🔥 Warm-up {i+1}/50 | Time: {elapsed:.3f} sec")

print("✅ Warm-up complete! Ready for real-time interaction.")

# 🎤 Real-Time Input Loop
while True:
    user_input = input("\n🤖 Enter prompt: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    start_time = time.time()
    full_prompt = system_prompt + "\n\n" + user_input  # 🔥 NO PREPROCESSING DELAY
    response = llm.generate(full_prompt, sampling_params)
    elapsed = time.time() - start_time

    print(f"\n💬 AI Response: {response[0].outputs[0].text.strip()}")
    print(f"⚡ Response Time: {elapsed:.3f} sec")