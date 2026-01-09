from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from embeddings.vector_store import *
from retriever.retriever import Retriever



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
    retriever = Retriever(embedder, store)

    # Query
    query = "Who is the Muhammad Ghilam Jillani?"
    results = retriever.retrieve_chunks(query, top_k=3)

    print("\n🔍 Top 3 Retrieved Chunks:\n")

    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(r["text"][:400])   # limit text length
        print("Metadata:", r["metadata"])
        print()
