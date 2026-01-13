from llm.hf_model import load_model
from llm.prompt_build import build_prompt
from llm.generator import generate_answer


class RAGPipeline():
    def __init__(self, retriever):
        self.retriever = retriever
        self.tokenizer, self.model = load_model()

    def run(self, query, top_k = 5):
        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve_chunks(query,
                                          top_k = top_k)
        
        # 2. Build prompt
        prompt = build_prompt(chunks, query)

        # 3. Generate Answer
        answer = generate_answer(
            prompt,
            self.tokenizer,
            self.model
        )

        return answer, chunks