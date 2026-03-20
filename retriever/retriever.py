class Retriever:
    """
    Retriever class for fetching relevant document chunks.

    This component connects:
    - The embedding model (to convert query → vector)
    - The vector store (to search similar chunks)

    Workflow:
    ---------
    1. Convert user query into embedding
    2. Perform similarity search in FAISS
    3. Return top-k relevant chunks

    This is a core part of the RAG pipeline.
    """

    def __init__(self, embedder, vector_store):
        """
        Initialize Retriever.

        Parameters
        ----------
        embedder : Embedder
            Instance of embedding model used to encode text.

        vector_store : FAISSStore
            Vector database storing document embeddings.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve_chunks(self, query: str, top_k: int = 5):
        """
        Retrieve top-k most relevant document chunks for a query.

        Parameters
        ----------
        query : str
            User input question.

        top_k : int, optional (default=5)
            Number of most relevant chunks to retrieve.

        Returns
        -------
        list of dict
            Retrieved chunks with:
            - "text": chunk content
            - "metadata": source information (page, file, etc.)

        Process:
        --------
        1. Convert query into embedding vector
        2. Search FAISS index using vector similarity
        3. Return best matching chunks
        """

        # Debug logs to verify correct object types
        print("DEBUG:")
        print("Embedder type:", type(self.embedder))
        print("Vector store type:", type(self.vector_store))

        # Step 1: Convert query into embedding
        query_embedding = self.embedder.embed_texts([query])[0]

        # Step 2: Retrieve top-k similar chunks from vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Step 3: Return retrieved chunks
        return results