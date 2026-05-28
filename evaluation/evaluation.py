# Import libraries
import json,os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv

# Import project module
from retriever.retriever import HybridRetriever
from reranker.reranker import Reranker
from llm.hf_model import load_llm
from pipelines.rag_pipelines import RAGPipeline
from embeddings.embedder import get_embedder
from utils.load_vector_store import vector_store_loader
from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents

# 1. Load Evaluation Dataset
with open("evaluation/sample_question.json", "r") as f:
    evaluation_data = json.load(f)

# 2. Build RAG Pipeline 
# Load documents
documents = pdf_loader("uploads/Chhabi_Lal_Tamang_Resume_ML.pdf")

# Create chunks
chunks = chunk_documents(documents)

# Load embeddings and vector store
embeddings = get_embedder()
vector_store = vector_store_loader(embeddings)

# Initialize Retriever
retriever = HybridRetriever(vector_store)

# Build sparse retriever
retriever.build_sparse_retriever(chunks)

# Initialize Reranker
reranker = Reranker()

# Initialize LLM
llm = load_llm()

# Build Pipeline
rag_pipeline = RAGPipeline(
    retriever=retriever,
    llm=llm,
    reranker=reranker
)


# 3. Run Evaluation Pipeline
rows = []

for item in evaluation_data:
    question = item['question']
    ground_truth = item['ground_truth']

    print(f"\n Evaluating Question: {question}")

    answer, retrieved_chunks = rag_pipeline.run(question)

    retrieved_contexts = [
        chunk.page_content
        for chunk in retrieved_chunks
    ] if retrieved_chunks else []

    rows.append({
        "user_input":question,
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truth

    })

dataset = Dataset.from_list(rows)

# 4. Load Environmen variables
load_dotenv()
MODEL_TOKEN = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(
    model = 'qwen/qwen3-32b',
    api_key = MODEL_TOKEN,
    temperature = 0
)
evaluator_llm = LangchainLLMWrapper(groq_llm)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embedding = LangchainEmbeddingsWrapper(embedding)

# 5. Run RAGAS Evaluation
print("\nRunning RAGAS Evaluation...")
print(type(Faithfulness()))

result = evaluate(
    dataset = dataset,
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm = evaluator_llm,
    embeddings=ragas_embedding
)

# Print results
print("\n ==== Evaluation Results ====")
print(result)

# Save result to json file
result_dict = result.to_pandas()
result_dict.to_json(
    "evaluation/results.json",
    orient = "records",
    indent = 4
)

