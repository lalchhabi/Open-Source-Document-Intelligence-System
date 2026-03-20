from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Embedding module for converting text into vector representations.

    This class uses a pre-trained SentenceTransformer model to generate
    dense vector embeddings for text. These embeddings are later stored
    in a vector database (FAISS) and used for similarity-based retrieval.

    Model Used:
    -----------
    Default: "BAAI/bge-small-en"
    - Lightweight and fast
    - Good performance for semantic search tasks

    Usage:
    ------
    embedder = Embedder()
    embeddings = embedder.embed_texts(["sample text"])
    """

    def __init__(self, model_name="BAAI/bge-small-en"):
        """
        Initialize the embedding model.

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained SentenceTransformer model to load.
            Default is "BAAI/bge-small-en".
        """
        # Load the pre-trained embedding model
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Generate vector embeddings for a list of text inputs.

        This function converts each text chunk into a numerical vector
        that captures its semantic meaning. These vectors are used for
        similarity search in the vector store.

        Parameters
        ----------
        texts : list of str
            List of text strings (chunks) to be embedded.

        Returns
        -------
        numpy.ndarray
            A 2D array where each row represents the embedding of a text.

        Example
        -------
        embeddings = embedder.embed_texts([
            "Machine learning is powerful",
            "RAG systems improve accuracy"
        ])
        """

        # Encode text into embeddings
        # show_progress_bar=True helps visualize progress for large datasets
        return self.model.encode(
            texts,
            show_progress_bar=True
        )

