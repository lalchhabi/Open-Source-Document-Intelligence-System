import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size = 400, chunk_overlap = 150):
    """
    Take the extracted documents from the pdf file and divide them into chunks
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