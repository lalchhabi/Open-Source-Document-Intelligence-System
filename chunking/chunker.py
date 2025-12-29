import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size = 500, chunk_overlap = 100):
    """
    Take the extracted documents from the pdf file and divide them into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = []

    for doc in documents:
        split_text = text_splitter.split_text(doc["text"])

        for i, chunk in enumerate(split_text):
            chunks.append({
                "text":chunk,
                "metadata":{
                    **doc["metadata"],
                    "chunk_id":1
                }

            })

    return chunks