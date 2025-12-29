from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents


if __name__ == "__main__":
    docs = pdf_loader("data/raw_docs")
    chunks = chunk_documents(docs)

    print(f"Total number of chunks created: {len(chunks)}\n")
    print("Sample chunk \n")
    print(chunks[3]["text"][:500])
    print("\nMetdata:", chunks[3]["metadata"])