# Import python libraries
from flask import Flask, request, render_template, jsonify
import os

# Import RAG pipeline libraries
from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from langchain_core.documents import Document

from embeddings.embedder import get_embedder
from embeddings.vector_store import create_vector_store

from retriever.retriever import HybridRetriever
from reranker.reranker import Reranker

from llm.hf_model import load_llm
from pipelines.rag_pipelines import RAGPipeline

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global RAG pipeline (shared across requests)
rag_pipeline = None

# Chat memory (stores last few conversations)
chat_history = []

# general chat model
llm = None

# Load general llm on start
def init_llm():
    global llm
    if llm is None:
        llm = load_llm()

    return llm



# ===============================
# MAIN PAGE ROUTE
# ===============================
@app.route("/", methods=["GET"])
def index():
    """
    Render main application UI page.

    This route only serves the frontend page.
    All interactions such as:
    - PDF upload
    - Question answering
    - Chat responses

    """

    return render_template("index.html")


    # Upload Document (Enable RAG mode)
@app.route("/upload", methods = ['POST'])
def upload():
    """Upload document and build RAG pipeline
    """

    global rag_pipeline, chat_history

    file = request.files.get("pdf")
    if not file:
        return jsonify({'status':'error', "msg":"No file uploaded"}),400
    
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    chat_history.clear()

    try:
        print("Loading PDF......")
        docs = pdf_loader(path)

        print("Chunking documents.......")
        chunks = chunk_documents(docs)
                                                                
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]

        print("Embedding............")
        embedder = get_embedder()

        documents =[
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metadata)
        ]

        print("Vector Store............")
        vector_store = create_vector_store(
            documents,
            embedder
        )

        print("Hybrid Retriever........")
        retriever = HybridRetriever(
            vector_store=vector_store,
            top_k=5
        )
        retriever.build_sparse_retriever(documents)

        print("LLM.............")
        llm = init_llm()

        print("RAG Pipeline......")
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            reranker=Reranker()
        )
        return jsonify({
            "status": "success",
            "msg": "Document Uploaded. RAG mode enabled."
        })
    
    except Exception as e:
        return jsonify({"status":'error', 'msg':str(e)}), 500
    
# Chat Endpoint (Smart Routing)
@app.route("/ask", methods = ['POST'])
def ask():
    """
    Handles both:
    1. General chat (LLM only)
    2. Document-based QA (RAG)
    """

    global rag_pipeline, chat_history

    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        # CASE 1: RAG MODE
        if rag_pipeline is not None:
            answer, sources = rag_pipeline.run(
                query=query,
                chat_history=chat_history
            )

            chat_history.append({
                "user": query,
                "assistant": answer
            })

            chat_history = chat_history[-5:]

            return jsonify({
                "mode": "rag",
                "answer": answer,
                "sources": [
                    {
                        "page": s.metadata.get("page", None),
                        "text": s.page_content[:200]
                    }
                    for s in sources
                ]
            })

        # CASE 2: NORMAL CHAT MODE
        llm = init_llm()
        response = llm.invoke(query)

        return jsonify({
            "mode": "chat",
            "answer": response.content if hasattr(response, "content") else response,
            "sources": []
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "msg": str(e)
        }), 500

# Run App
if __name__ == "__main__":
    app.run(debug = True, use_reloader = False)