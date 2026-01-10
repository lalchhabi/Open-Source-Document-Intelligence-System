class Retriever:
    def __init__(self, embedder, vector_store):
        """
        embedder: instance of Embedder vector_store: FAISStore instance
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve_chunks(self,query: str, top_k: int=5):
        """
        Embed the query and retrieve top-k relevant chunks
        """
        query_embedding = self.embedder.embed_texts([query])[0]

        results = self.vector_store.search(
            query_embedding,
            top_k
        )
        return results