### Import libraries
from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import get_embedder
from langchain_core.documents import Document
from embeddings.vector_store import create_vector_store
from retriever.retriever import HybridRetriever
from reranker.reranker import Reranker
from llm.hf_model import load_llm
from pipelines.rag_pipelines import RAGPipeline


def main():
    print("Starting RAG System........")

    # Step 1: Load documents
    docs = pdf_loader("data/raw_docs/offer_letter_chhabi.pdf")

    # Step 2: Chunk documents
    chunks = chunk_documents(docs)

    # Step 3: Prepare texts and metadata
    texts = [c.page_content for c in chunks]
    metadata = [c.metadata for c in chunks]

    # Step 4: Load Embedding Model
    embedder = get_embedder()

    # Step 5: Convert to langchain documents
    documents = [
        Document(
            page_content = text,
            metadata = meta
        )
        for text, meta in zip(texts, metadata)
    ]

    # Step 6: Create Vector Store
    vector_store = create_vector_store(
        documents,
        embedder
    )

    # Step 7: User Query
    query = (
        "Based on the contract paper Can you tell the position name of employee?"
    )

    # Step 8: Create Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        top_k=5
    )

    # Build BM25 sparse retriever
    hybrid_retriever.build_sparse_retriever(
        documents
    )

    # Step 9: Initialize Reranker
    reranker = Reranker()

    # Step 10: Load LLM model
    llm = load_llm()

    # Step 11: Initialize RAG pipeline 
    rag = RAGPipeline(
        retriever=hybrid_retriever,
        llm=llm,
        reranker = reranker,
        
    )

    # Step 12: Run RAG pipeline
    answer, sources = rag.run(query)

    # Step 13: Output Response
    print("\n====================")
    print("ANSWER:\n")
    print(answer)

    print("\n====================")
    print("Sources: \n")

    for i, doc in enumerate(sources, start=1):
        print(f"{i}. {doc.metadata}")

if __name__ == "__main__":
    main()
    