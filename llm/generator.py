from llm.hf_model import *
import torch 

def generate_answer(prompt, max_tokens = 300):
    inputs  = tokenizer(prompt, return_tensors = "pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            temperature = 0.2,
            top_p = 0.9,
            do_sample = True
        )

    answer = tokens.decode(
        outputs[0], skip_special_tokens = True
    )
    return answer