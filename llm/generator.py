from llm.hf_model import *
import torch 

def generate_answer(prompt, tokenizer, model,  max_tokens=150):

    print("🧠 Generating answer...")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = generated_text[len(prompt):].strip()

    print("✅ Done")
    return answer
