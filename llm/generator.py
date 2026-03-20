from llm.hf_model import *
import torch 


def generate_answer(prompt, tokenizer, model, max_tokens=150):
    """
    Generate an answer from the LLM using the given prompt.

    This function takes a formatted prompt (including context and query),
    feeds it into a HuggingFace model, and returns the generated answer.

    Parameters
    ----------
    prompt : str
        Final prompt containing context + user query.

    tokenizer : HuggingFace Tokenizer
        Tokenizer used to convert text → tokens.

    model : HuggingFace Model
        Pre-trained language model used for generation.

    max_tokens : int, optional (default=150)
        Maximum number of tokens to generate for the answer.

    Returns
    -------
    str
        Generated answer (cleaned, without prompt text).

    Process:
    --------
    1. Tokenize input prompt
    2. Run model generation
    3. Decode output tokens to text
    4. Remove prompt from generated text
    """

    print("Generating answer...")

    # Step 1: Convert prompt into model input tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,      # Prevent overflow
        max_length=2048       # Limit input size (model constraint)
    ).to(model.device)

    # Step 2: Generate output tokens (inference mode)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,   # Limit answer length
            temperature=0.3,             # Lower = more deterministic
            top_p=0.9,                   # Nucleus sampling
            do_sample=True,              # Enable sampling
            eos_token_id=tokenizer.eos_token_id  # Stop at end token
        )

    # Step 3: Convert tokens → readable text
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Step 4: Remove prompt from output to get only answer
    answer = generated_text[len(prompt):].strip()

    print("Done")

    return answer