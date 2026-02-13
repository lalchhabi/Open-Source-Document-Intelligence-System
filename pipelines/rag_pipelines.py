from llm.hf_model import load_model
from llm.prompt_build import build_prompt
from llm.generator import generate_answer


class RAGPipeline():

    def __init__(self, retriever):
        self.retriever = retriever
        self.tokenizer, self.model = load_model()

    def run(self, query, top_k=5, chat_history=None):

        print("\n🔍 Retrieving chunks...")
        chunks = self.retriever.retrieve_chunks(query, top_k)

        print("📄 Building prompt...")
        prompt = build_prompt(
            chunks,
            query,
            chat_history
        )

        print("🚀 Sending to LLM...")
        answer = generate_answer(
            prompt,
            self.tokenizer,
            self.model
        )

        print("✅ Answer generated")

        return answer, chunks
