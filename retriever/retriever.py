### Import libraries
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

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
        top_k: int
            Number of documents to return
        """
        self.vector_store = vector_store
        self.top_k = top_k

        ### Dense retriever (semantic search FAISS)
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

    def retrieve(self, query, dense_weight=0.7, sparse_weight=0.3):
        """
        Combine dense and sparse retrievers into a hybrid retriever.
        
        Parameters
        ----------
        query : str
        dense_weight : float
        sparse_weight : float

        Returns
        -------
        list[Document]
        """

        if self.sparse_retriever is None:
            raise ValueError("Sparse retriever not intialized. Call build_sparse)retriever() first.")
        
        # Step 1: Retrieve from both systems
        dense_docs = self.dense_retriever.invoke(query)

        sparse_docs = self.sparse_retriever.invoke(query)

        # Step 2: Score fusion using dictionary
        doc_scores = {}

        for doc in dense_docs:
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0) + dense_weight
        
        for doc in sparse_docs:
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0) + sparse_weight

        # Step 3: Convert back to Document objects
        all_docs = {doc.page_content: doc for doc in dense_docs + sparse_docs}

        # Step 4: Sort by score
        sorted_docs = sorted(
            doc_scores.items(),
            key = lambda x: x[1],
            reverse=True
        )
        
        # Step 5: Build final ranked list
        final_docs = [
            all_docs[text]
            for text, _ in sorted_docs
        ]

        return final_docs[:self.top_k]