### import libraries
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

### Load environment variables 
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

### Load model name
MODEL_NAME = 'google/gemma-2-2b-it'

def load_llm():
    """Load Hugging Face LLM via Inference API
    """
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        )
    
    chat_model = ChatHuggingFace(llm = llm)
    
    return chat_model

