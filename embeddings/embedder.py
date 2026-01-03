from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)
        pass

    def embed_texts(self,texts):
        return self.model.encode(texts, 
                                show_progress_bar=True)

