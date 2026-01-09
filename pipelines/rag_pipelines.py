from llm.hf_model import load_llm
from llm.prompt_build import build_prompt
from llm.generator import generate_answer
from retriever.retriever import Retriever


retriever = Retriever
retrieve_chunks = retriever.retrieve_chunks
tokenizer, model = load_llm()

def run_rag(query, vector_store):
    chunks = retrieve_chunks(query,vector_store, top_k=5)
    prompt = build_prompt(chunks, query)
    answer = generate_answer(prompt, tokenizer, model)
    return answer, chunks
