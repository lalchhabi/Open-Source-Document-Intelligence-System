### import libraries
from llm.hf_model import load_llm
from langchain_core.messages import HumanMessage

class Generator:
    """
    Handles response generation using Langchain LLM.
    """

    def __init__(self):
        self.llm = load_llm()

    def generate_answer(self, prompt):
        """Generate answer using Langchain LLM.

        Parameters
        Args:
            prompt (str): Final formatted prompt from RAG pipeline.

        Returns
            Generated response from LLM.
        """

        print("Generating Answer......")
        response = self.llm.invoke([
            HumanMessage(content=prompt)
            ])
        print("Done")
        return response.content