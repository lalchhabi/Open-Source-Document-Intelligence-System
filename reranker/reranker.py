### Import libraries
import os
import cohere
from dotenv import load_dotenv
from langsmith import traceable


# Load environment variables
load_dotenv()

class Reranker:
    """
    Reranks retrieved documents using Cohere's hosted reranking API.
    """

    def __init__(
            self,
            model_name="rerank-v3.5",
            top_k = 5
            ):
        
        """
        Initializes the Cohere client using the API key stored in the environment variables.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.client = cohere.ClientV2(
            api_key=os.getenv("COHERE_API_KEY")
        )

    @traceable(name="cohere_reranker", run_type="chain")
    def rerank(self, query, documents, top_k=None):
        """
         Reranks retrieved documents based on their semantic relevance
        to the user query.

        Parameters
        ----------
        query : str
            User's search query.

        documents : list
            List of retrieved LangChain Document objects.

        top_k : int, default=self.top_k
            Number of highest-ranked documents to return.

        Returns
        -------
        list
            Top-k LangChain Document objects ranked by the
            Cohere reranking model.
        """
        # Extract plain text from LangChain Document objects
    
        doc_texts = [
            doc.page_content for doc in documents
        ]

        # Send query and documents to Cohere's reranking API
        response = self.client.rerank(
            model = self.model_name,
            query = query,
            documents = doc_texts,
            top_n = top_k if top_k is not None else self.top_k
        )

        # Reconstruct the ranked LangChain Document objects
        ranked_docs = [
            documents[result.index] for result in response.results
        ]

        return ranked_docs
    
