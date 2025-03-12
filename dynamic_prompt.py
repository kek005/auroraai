from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 🔥 Load Small Open-Source Model for Prompt Rewriting
rewrite_model = "google/t5-small"
tokenizer = AutoTokenizer.from_pretrained(rewrite_model)
model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model)

def reword_question(question):
    """ Convert open-ended question into a direct command """
    input_text = f"Rewrite this question as a direct command: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🎤 Example:
question = "What are the benefits of using Azure Virtual Machines?"
print("🔹 Rewritten Prompt:", reword_question(question))