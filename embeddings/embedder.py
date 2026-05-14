### Import libraries
from langchain_huggingface import HuggingFaceEmbeddings, 

def get_embedder(
    model_name = "BAAI/bge-small-en",
    normalize = True,
    device = None
):
    """
    Creates a LangChain-compatible embedding model.

    This uses HuggingFaceEmbeddings wrapper which integrates directly with:
    - FAISS
    - Chroma
    - LangChain retrievers
    - LangSmith tracing

    Parameters
    ----------
    model_name : str
        HuggingFace SentenceTransformer model name.

    normalize : bool
        Whether to normalize embeddings (important for cosine similarity search).

    device : str or None
        Device to run model on ("cpu", "cuda"). If None, auto-detects.

    Returns
    -------
    HuggingFaceEmbeddings
        LangChain embedding object ready for vector stores.
    """
    model_kwargs = []
    if device:
        model_kwargs['device'] = device

    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = {
            "normalize_embeddings": normalize
        }
    )
    return embeddings