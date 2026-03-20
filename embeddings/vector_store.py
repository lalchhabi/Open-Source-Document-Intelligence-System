import faiss
import numpy as np


class FAISSStore:
    """
    FAISS-based Vector Store for efficient similarity search.

    This class stores text embeddings and enables fast nearest neighbor
    search using Facebook AI Similarity Search (FAISS).

    Key Responsibilities:
    ---------------------
    1. Store embeddings in FAISS index
    2. Maintain mapping between embeddings and original texts
    3. Retrieve most relevant text chunks based on query similarity

    Distance Metric:
    ----------------
    Uses L2 (Euclidean distance) for similarity search.

    Usage:
    ------
    store = FAISSStore(embedding_dim=384)
    store.add(embeddings, texts, metadata)
    results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, embedding_dim):
        """
        Initialize FAISS index.

        Parameters
        ----------
        embedding_dim : int
            Dimension of embedding vectors (must match model output size).
        """

        # Create FAISS index using L2 distance (Euclidean)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Store original texts and metadata (parallel to embeddings)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadata):
        """
        Add embeddings and corresponding data to the vector store.

        Parameters
        ----------
        embeddings : list or numpy.ndarray
            List of embedding vectors.

        texts : list of str
            Original text chunks corresponding to embeddings.

        metadata : list of dict
            Metadata for each text chunk (e.g., source file, page number).

        Notes
        -----
        - Embeddings must be converted to float32 for FAISS compatibility.
        - Order must be preserved between embeddings, texts, and metadata.
        """

        # Convert embeddings to NumPy float32 array (required by FAISS)
        self.index.add(np.array(embeddings).astype("float32"))

        # Store text and metadata in same order as embeddings
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=8):
        """
        Retrieve top-k most similar text chunks for a query.

        Parameters
        ----------
        query_embedding : numpy.ndarray
            Embedding vector of the query.

        top_k : int, optional (default=8)
            Number of most relevant chunks to retrieve.

        Returns
        -------
        list of dict
            List of retrieved chunks, each containing:
            - "text": relevant text chunk
            - "metadata": associated metadata

        Example
        -------
        results = store.search(query_embedding, top_k=5)
        """

        # Perform similarity search in FAISS
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        results = []

        # Map retrieved indices back to original texts and metadata
        for idx in indices[0]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx]
            })

        return results