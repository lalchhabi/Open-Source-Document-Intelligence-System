### Import libraries
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever

class HybridRetriever:
    """
    Hybrid Retriever for RAG systems combining semantic and keyword search.

    Combines two retrieval methods:
    1. Dense Retrieval (FAISS)
       - Uses embeddings for semantic similarity
       - Captures meaning-based matches

    2. Sparse Retrieval (BM25)
       - Uses keyword matching
       - Strong for exact terms, names, and numbers

    Workflow:
    Query -> Dense search + Sparse search -> Weighted combination -> Top results

    Output:
    -------
    Returns most relevant document chunks for RAG pipeline.
    """

    def __init__(self, vector_store, top_k=5):
        """
        Initialize Retriever.

        Parameters
        vector_store : FAISS
            Langchain FAISS vector store.
        """
        self.vector_store = vector_store
        self.top_k = top_k

        ### Dense retriever (semantic search)
        self.dense_retriever = vector_store.as_retriever(
            search_kwargs={'k':self.top_k}
        )

        ### Sparse retriever (keyword search)
        self.sparse_retriever = None

    def build_sparse_retriever(self, documents):
        """
        Create BM25 retriever from documents
        """
        self.sparse_retriever = BM25Retriever.from_documents(
            documents
        )
        self.sparse_retriever.k = self.top_k

    def get_retriever(self):
        """
        Combine dense and sparse retrievers into a hybrid retriever.
        
        Returns:
            EnsembleRetriever (hybrid retriever object)
        """

        ensemble_retriever = EnsembleRetriever(
            retrievers = [
                self.dense_retriever,
                self.sparse_retriever
            ],
            weights = [0.7, 0.3] ### assign priority
        )
        return ensemble_retriever