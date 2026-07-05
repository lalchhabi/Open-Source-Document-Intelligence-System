### Import libraries
from langchain_community.embeddings import FastEmbedEmbeddings

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedder(
    model_name=DEFAULT_EMBEDDING_MODEL,
):
    """
    Creates a LangChain-compatible FastEmbed embedding model.

    This implementation uses FastEmbed, an ONNX-based embedding library
    optimized for fast CPU inference with a smaller runtime footprint
    compared to the standard Hugging Face SentenceTransformers backend.

    The returned embedding model integrates directly with:
    - FAISS
    - Chroma
    - LangChain retrievers
    - LangSmith tracing

    Parameters
    ----------
    model_name : str
        Name of the FastEmbed-compatible embedding model.
        Defaults to "BAAI/bge-small-en-v1.5".

    Returns
    -------
    FastEmbedEmbeddings
        LangChain-compatible embedding model ready for generating
        document and query embeddings.
    """

    embeddings = FastEmbedEmbeddings(
        model_name=model_name
    )

    return embeddings