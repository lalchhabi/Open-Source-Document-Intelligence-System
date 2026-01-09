from llm.hf_model import *
import torch 

def generate_answer(prompt, max_tokens = 300):
    inputs  = tokenizer(prompt, return_tensors = "pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            
        )