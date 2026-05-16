### Import libraries
from langchain_community.vectorstores import FAISS

class FAISSVectorStore:
    """
    LangChain-based FAISS vector store for semantic retrieval.

    This class handles:
    - document embedding
    - vector indexing
    - similarity search
    - metadata storage

    using LangChain's FAISS integration.
    """

    def __init__(self, embeddings):
        """
        Initialize vector store.

        Parameters
        ----------
        embeddings :
            LangChain embedding model.
        """
        self.embeddings = embeddings
        self.vector_store = None

    def create_vector_store(self, documents):
        """
        Create FAISS vector store from chunked documents.

        Parameters
        ----------
        documents : list[Document]
            Chunked LangChain documents.
        """

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

    def similarity_search(self, query, top_k=5):
        """
        Retrieve most relevant chunks for a query.

        Parameters
        ----------
        query : str
            User query.

        top_k : int
            Number of documents to retrieve.

        Returns
        -------
        list[Document]
            Retrieved relevant documents.
        """

        return self.vector_store.similarity_search(
            query=query,
            k=top_k
        )