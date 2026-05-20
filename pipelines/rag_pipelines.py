### Import libraries
from llm.prompt_build import build_prompt
from reranker.reranker import Reranker

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.

    This pipeline orchestrates:
    - document retrieval
    - context construction
    - prompt formatting
    - LLM-based answer generation

    using LangChain components and LCEL.
    """

    def __init__(self, retriever, llm, reranker):
        """
        Initialize RAG Pipeline.

        Parameters
        ----------
        retriever : Retriever responsible for fetching relevant documents.
        llm : LangChain chat model used for response generation.
        """

        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

        # Build prompt template
        self.prompt = build_prompt()

        # Langchain Expression Language
        self.chain = self.prompt | self.llm

    def run(self, query, retrieve_k=10, top_k=5, chat_history=""):
        """
        Execute the full RAG pipeline.

        Parameters
        ----------
        query : str
            User question.

        top_k : int, optional (default=5)
            Number of relevant chunks to retrieve.

        chat_history : list, optional
            Previous conversation history for context-aware responses.

        Returns
        -------
        tuple
            answer : str
                Generated response from the LLM.

            chunks : list
                Retrieved document chunks used as context.

        Steps:
        ------
        1. Retrieve relevant chunks from vector store
        2. Build prompt using retrieved context + query + history
        3. Generate answer using LLM
        """

        # Step 1: Retrieve relevant chunks
        print("\nRetrieving chunks...")
        chunks = self.retriever.retrieve(
            query
            )
        
        # Step 2: Implement Reranker in retrieve chunks
        reranked_chunks = self.reranker.rerank(
            query=query,
            documents=chunks,
            top_k=top_k
        )

        # Step 3: Convert chunks into context string
        context = "\n\n".join([
            chunk.page_content
            for chunk in reranked_chunks
        ])

        print("Generating answer...")

        # Step 3: Generate answer using LCEL chain
        response  = self.chain.invoke({
            'context': context,
            'question': query,
            'chat_history': chat_history
        })

        print("Answer generated")

        # Return both answer and sources (chunks)
        return response.content, reranked_chunks