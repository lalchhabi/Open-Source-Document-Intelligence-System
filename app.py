from flask import Flask, request, render_template, jsonify
import os
from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from embeddings.vector_store import FAISSStore
from retriever.retriever import Retriever
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


# ===============================
# MAIN PAGE ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Render main UI page.

    Handles:
    - Page load (GET)
    - Basic form-based query (POST) [optional fallback]

    Note:
    Most interactions now happen via AJAX (/ask endpoint).
    """

    global rag_pipeline

    answer = None
    sources = []
    message = None

    if request.method == "POST":

        # Get user query from form
        query = request.form.get("query")

        if query and rag_pipeline:
            # Run RAG pipeline
            answer, sources = rag_pipeline.run(query)

        elif query and not rag_pipeline:
            message = "⚠️ Please upload a document first."

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        message=message
    )


# ===============================
# UPLOAD ROUTE (RAG PIPELINE BUILD)
# ===============================
@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle PDF upload and build full RAG pipeline.

    Steps:
    1. Save uploaded file
    2. Load PDF text
    3. Chunk documents
    4. Generate embeddings
    5. Store in FAISS
    6. Initialize retriever
    7. Create RAG pipeline

    Returns
    -------
    JSON response indicating success or error
    """

    global rag_pipeline, chat_history

    file = request.files.get("pdf")

    if not file:
        return jsonify({"status": "error", "msg": "No file"}), 400

    # Save uploaded file
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Clear chat history when new document is uploaded
    chat_history.clear()

    try:
        print("📄 Loading PDF...")
        docs = pdf_loader(path)

        print("✂️ Chunking...")
        chunks = chunk_documents(docs)

        # Extract text and metadata
        texts = [c["text"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]

        print("🧠 Embedding...")
        embedder = Embedder()
        embeddings = embedder.embed_texts(texts)

        print("💾 Creating FAISS store...")
        store = FAISSStore(len(embeddings[0]))
        store.add(embeddings, texts, metadata)

        print("🔍 Creating Retriever...")
        retriever = Retriever(embedder, store)

        print("🤖 Creating RAG pipeline...")
        rag_pipeline = RAGPipeline(retriever)

        print("✅ Pipeline Ready!")

        return jsonify({"status": "success"})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"status": "error", "msg": str(e)}), 500


# ===============================
# QUESTION ANSWERING ROUTE
# ===============================
@app.route("/ask", methods=["POST"])
def ask():
    """
    Handle user queries (AJAX-based).

    Steps:
    1. Receive query from frontend
    2. Run RAG pipeline
    3. Update chat memory
    4. Return answer + sources

    Returns
    -------
    JSON:
        {
            "answer": str,
            "sources": list
        }
    """

    global chat_history, rag_pipeline

    data = request.get_json()
    query = data["query"]

    # Ensure pipeline is ready
    if not rag_pipeline:
        return jsonify({"answer": "Upload document first."})

    # Run RAG pipeline with memory
    answer, sources = rag_pipeline.run(
        query=query,
        top_k=5,
        chat_history=chat_history
    )

    # Save conversation to memory
    chat_history.append({
        "user": query,
        "assistant": answer
    })

    # Keep only last 5 conversations
    del chat_history[:-5]

    # Return response + sources
    return jsonify({
        "answer": answer,
        "sources": [
            {
                "page": s["metadata"]["page"],
                "text": s["text"][:200]  # preview only
            }
            for s in sources
        ]
    })


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)