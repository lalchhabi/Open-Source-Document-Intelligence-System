from langchain_community.vectorstores import FAISS

def vector_store_loader(embeddings):
    """This function actually load vector store that are saved locally
    """
    vector_store = FAISS.load_local(
        "embeddings/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store