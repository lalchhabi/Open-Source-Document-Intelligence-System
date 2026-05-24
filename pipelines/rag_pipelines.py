### Import libraries
from llm.prompt_build import build_prompt
from reranker.reranker import Reranker
from langsmith import traceable

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

    @traceable(name="rag_pipeline")
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
        chunks, relevance_score = self.retriever.retrieve(query)

        # Threshold 
        THRESHOLD = 0.3

        if relevance_score < THRESHOLD:
            return None, None  # means "use normal chat"
        
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
    
    def stream(self, query, top_k=5, chat_history = ""):
        """Stream responses from the Retrieval-Augmented Generation (RAG) pipeline.

        This function executes the full RAG workflow incrementally and yields
        partial LLM outputs in real-time instead of waiting for the full response.


        Args:
            query (str): user question or input prompt
            retrieve_k (int, optional): Number of documents initially retrieved. Defaults to 10.
            top_k (int, optional): Top ranked documents. Defaults to 5.
            chat_history (str, optional): Previous conversation history used to maintain context. Defaults to "".
        """
        print("\n Retrieving Chunks.....")

        chunks, relevance_score = self.retriever.retrieve(
            query=query,
        )

        THRESHOLD = 0.3
        
        # Router Logic
        if relevance_score < THRESHOLD:
            print("Switching to Normal Chat Mode")
            return self.llm.stream(query), []
        
        

        reranked_chunks = self.reranker.rerank(
            query = query,
            documents = chunks,
            top_k=top_k
        )

        context = "\n\n".join([
            chunk.page_content
            for chunk in reranked_chunks
        ])

        print("Streaming Answer.........")

        stream_response = self.chain.stream({
            "context": context,
            "question": query,
            "chat_history": chat_history
        })

        return stream_response, reranked_chunks