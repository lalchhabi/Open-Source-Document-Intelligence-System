from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=900, chunk_overlap=200):
    """
    Split documents into smaller overlapping text chunks.

    This function takes a list of cleaned documents (from the loader)
    and divides each document into smaller pieces using a recursive
    text splitter. Chunking is essential for RAG systems because
    embedding models perform better on shorter, meaningful text segments.

    Parameters
    ----------
    documents : list of dict
        List of documents where each document contains:
        - "text": cleaned text
        - "metadata": information like source file and page number

    chunk_size : int, optional (default=900)
        Maximum number of characters per chunk.

    chunk_overlap : int, optional (default=200)
        Number of overlapping characters between consecutive chunks.
        This helps preserve context across chunk boundaries.

    Returns
    -------
    list of dict
        A list of chunk dictionaries where each chunk contains:
        - "text": chunked text
        - "metadata": original metadata + chunk_id

    """

    # Initialize LangChain recursive text splitter
    # It splits text based on hierarchy: paragraphs → lines → sentences → words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # priority of splitting
    )

    chunks = []

    # Loop through each document
    for doc in documents:

        # Split document text into chunks
        split_texts = text_splitter.split_text(doc["text"])

        # Assign chunk IDs and attach metadata
        for i, chunk in enumerate(split_texts):

            # Filter out very small or weak chunks
            # (helps improve embedding and retrieval quality)
            if len(chunk.strip()) > 200:

                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],   # keep original metadata
                        "chunk_id": i        # add chunk identifier
                    }
                })

    return chunks