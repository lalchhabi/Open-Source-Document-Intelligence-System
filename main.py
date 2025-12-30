from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore



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

    # Query
    query = "What is the leave policy for contract employees?"
    query_embedding = embedder.embed_texts([query])[0]
    results = store.search(query_embedding)

    print("\n Top Retrieved Chunks: \n")
    for r in results:
        print(r['text'][:300])
        print("Metadata:", r['metadata'])
        print("-----"*30)