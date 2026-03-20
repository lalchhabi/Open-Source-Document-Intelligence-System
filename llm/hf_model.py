from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

# Pre-trained model name (instruction-tuned model)
model_name = "google/gemma-2b-it"


def load_model():
    """
    Load HuggingFace tokenizer and language model.

    This function initializes the LLM used in the RAG pipeline.
    It loads both:
    - Tokenizer → converts text into tokens
    - Model → generates responses from tokens

    Returns
    -------
    tuple
        tokenizer : AutoTokenizer
            Tokenizer for encoding/decoding text.

        model : AutoModelForCausalLM
            Pre-trained causal language model for text generation.

    Notes
    -----
    - Uses half precision (float16) for faster inference
    - Uses automatic device mapping (CPU/GPU)
    - Optimized for local LLM inference
    """

    # Load tokenizer (text → tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # Use FP16 for faster performance
        device_map="auto"            # Automatically use GPU if available
    )

    return tokenizer, model