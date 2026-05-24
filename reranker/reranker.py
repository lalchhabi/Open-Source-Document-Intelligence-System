### Import libraries
from sentence_transformers import CrossEncoder
from langsmith import traceable

class Reranker:
    """
    Reranks retrieved documents using cross-encoder model.
    """

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base"
    ):

        self.model = CrossEncoder(model_name)

    @traceable(name="cross_encoder_reranker", run_type="chain")
    def rerank(self, query, documents, top_k=3):
        """
        Rerank retrieved documents.
        """

        # Create query-document pairs
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Combine docs + scores
        scored_docs = list(zip(documents, scores))

        # Sort by score descending
        ranked_docs = sorted(
            scored_docs,
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k documents
        return [
            doc for doc, score in ranked_docs[:top_k]
        ]
    
