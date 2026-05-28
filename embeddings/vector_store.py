### Import libraries
from langchain_community.vectorstores import FAISS

def create_vector_store(documents, embeddings):
    """
    Create FAISS vector store using LangChain.

    Parameters                                                                        
    ----------
    documents : list
        Chunked documents (LangChain Document objects)

    embeddings :
        LangChain embedding model

    Returns
    -------
    FAISS vector store object
    """   

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Save locally
    vector_store.save_local("embeddings/faiss_index")

    return vector_store
