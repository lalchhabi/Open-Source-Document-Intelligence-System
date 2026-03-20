from llm.hf_model import load_model
from llm.prompt_build import build_prompt
from llm.generator import generate_answer


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Pipeline.

    This class orchestrates the full pipeline:
    1. Retrieve relevant document chunks
    2. Build a structured prompt
    3. Generate answer using LLM

    Components:
    -----------
    retriever : Retriever
        Handles semantic search over document embeddings.

    tokenizer, model :
        HuggingFace model used for text generation.

    Workflow:
    ---------
    User Query → Retrieve Chunks → Build Prompt → Generate Answer
    """

    def __init__(self, retriever):
        """
        Initialize RAG Pipeline.

        Parameters
        ----------
        retriever : Retriever
            Instance responsible for retrieving relevant chunks.
        """

        # Retriever handles vector search
        self.retriever = retriever

        # Load LLM model + tokenizer
        self.tokenizer, self.model = load_model()

    def run(self, query, top_k=5, chat_history=None):
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
            Format:
            [
                {"user": "...", "assistant": "..."},
                ...
            ]

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
        chunks = self.retriever.retrieve_chunks(query, top_k)

        # Step 2: Build prompt
        print("Building prompt...")
        prompt = build_prompt(
            chunks,
            query,
            chat_history
        )

        # Step 3: Generate answer using LLM
        print("Sending to LLM...")
        answer = generate_answer(
            prompt,
            self.tokenizer,
            self.model
        )

        print("Answer generated")

        # Return both answer and sources (chunks)
        return answer, chunks