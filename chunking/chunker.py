from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    """
    Splits input documents into smaller, token-aware chunks for use in a RAG (Retrieval-Augmented Generation) system.

    This function improves retrieval quality by breaking large documents into smaller,
    semantically manageable pieces that fit within LLM token limits.
    
    It uses a hybrid strategy:
    - Structure-aware splitting (paragraphs, lines, sentences)
    - Token-aware chunk sizing using tiktoken encoder
    - Overlapping chunks to preserve context across boundaries
    - Basic filtering to remove noisy or too-small chunks

    Parameters
    ----------
    documents : list[Document]
        List of LangChain Document objects containing page_content and metadata.

    chunk_size : int, default=300
        Maximum number of tokens allowed per chunk.
        Controls the granularity of retrieval (smaller = more precise, larger = more context).

    chunk_overlap : int, default=50
        Number of overlapping tokens between consecutive chunks.
        Helps maintain context continuity across chunk boundaries.

    Returns
    -------
    list[Document]
        A list of processed document chunks ready for embedding and vector storage.
        Each chunk contains:
        - page_content (chunk text)
        - metadata (source, page, chunk_id, etc.)
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ]
    )

    chunks = splitter.split_documents(documents)

    filtered_chunks = []

    for i, chunk in enumerate(chunks):

        text = chunk.page_content.strip()

        if len(text) > 200 and len(text.split()) > 40:

            chunk.metadata["chunk_id"] = i
            filtered_chunks.append(chunk)

    return filtered_chunks