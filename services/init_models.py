# import libraries
from embeddings.embedder import get_embedder
from reranker.reranker import Reranker
from llm.hf_model import load_llm


class ModelRegistry:
    """Loads heavy models Once at startup
    """
    def __init__(self):
        
        print("Loading Embedder............")
        self.embedder = get_embedder()

        print("Loading Reranker............")
        self.reranker = Reranker()

        print("Loading LLM............")
        self.llm = load_llm()

        print("All models loaded successfully ")

# Singleton instance
model_registry = ModelRegistry()