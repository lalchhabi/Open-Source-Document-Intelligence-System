### import libraries
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

### Load environment variables 
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

### Load model name
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'

def load_llm():
    """Load Hugging Face LLM via Inference API
    """
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        task = "conversational",
        temperature=0.3,
        huggingfacehub_api_token=HF_TOKEN,
        streaming=True
        )
    
    model = ChatHuggingFace(llm = llm)
    
    return model

