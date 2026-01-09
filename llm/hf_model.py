from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

model_name = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map = "auto"
)