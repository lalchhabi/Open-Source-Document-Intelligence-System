import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size = 900, chunk_overlap = 200):
    """
    Split long documents into smaller overlapping chunks.

    This function takes the cleaned documents produced by the
    PDF loader and divides them into smaller segments. Chunking
    improves retrieval accuracy because embedding models work
    better with shorter text segments.

    Parameters
    ----------
    docs : list
        List of dictionaries containing document text and metadata.

    chunk_size : int
        Maximum number of characters per chunk.

    overlap : int
        Number of characters shared between consecutive chunks to
        preserve context across chunk boundaries.

    Returns
    -------
    list
        List of chunk dictionaries with text and metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
         separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for doc in documents:
        for i, chunk in enumerate(text_splitter.split_text(doc["text"])):
            if len(chunk.strip()) > 200:   # filter weak chunks
                chunks.append({
                    "text": chunk,
                    "metadata": {**doc["metadata"], "chunk_id": i}
                })
    return chunks

