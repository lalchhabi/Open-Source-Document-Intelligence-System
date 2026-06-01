# Import libraries
from embeddings.vector_store import create_vector_store
from retriever.retriever import HybridRetriever
from pipelines.rag_pipelines import RAGPipeline

class RAGService:
    """
    Singleton service that manages all RAG components.
    Keeps system stable across Flask + Docker + requests.
    """

    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.rag_pipeline = None
        self.mode = "chat"

    def build(self, documents, embeddings, llm, reranker):
        """
        Build full RAG system once when document is uploaded.
        """

        print("\n[ RAG SERVICE ] Building vector store...")
        self.vector_store = create_vector_store(documents, embeddings)

        print("[ RAG SERVICE ] Building retriever...")
        self.retriever = HybridRetriever(self.vector_store)
        print("[ RAG SERVICE ] Sparse retriever...")
        self.retriever.build_sparse_retriever(documents)

        print("[ RAG SERVICE ] Creating reranker...")
        print("RERANKER TYPE:", type(reranker))


        print("[ RAG SERVICE ] Building pipeline...")
        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            llm=llm,
            reranker=reranker
        )

        self.mode = "rag"
        print("[ RAG SERVICE ] READY ")



    def reset(self):
        """
        Switch back to chat mode (remove document context)
        """
        self.vector_store = None
        self.retriever = None
        self.rag_pipeline = None
        self.mode = "chat"

    def get_pipeline(self):
        return self.rag_pipeline