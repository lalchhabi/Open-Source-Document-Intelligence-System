from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from embeddings.vector_store import *
from retriever.retriever import Retriever
from pipelines.rag_pipelines import RAGPipeline


if __name__ == "__main__":
    docs = pdf_loader("data/raw_docs")
    chunks = chunk_documents(docs)

    texts = [c['text'] for c in chunks]
    metadata = [c['metadata'] for c in chunks]

    # Embed
    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)

    # Store in FAISS
    store = FAISSStore(embedding_dim=len(embeddings[0]))
    store.add(embeddings, texts, metadata)

    # Create retriever
    retriever = Retriever(
    embedder=embedder,
    vector_store=store)


    # Query
    query = "Based on the contract paper can you tell the employee name."

    # Create RAG Pipeline object
    rag = RAGPipeline(retriever)
    
    answer, sources = rag.run(
        query
    )

    print(f"Answer: {answer}")