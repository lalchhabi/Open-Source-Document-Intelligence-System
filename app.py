from flask import Flask, request, render_template, jsonify
import os

from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from embeddings.vector_store import FAISSStore
from retriever.retriever import Retriever
from pipelines.rag_pipelines import RAGPipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global pipeline
rag_pipeline = None


# ===============================
# MAIN PAGE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    global rag_pipeline

    answer = None
    sources = []
    message = None

    if request.method == "POST":

        # QUESTION HANDLING
        query = request.form.get("query")

        if query and rag_pipeline:
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
# UPLOAD ROUTE (REAL PIPELINE)
# ===============================
@app.route("/upload", methods=["POST"])
def upload():
    global rag_pipeline

    file = request.files.get("pdf")

    if not file:
        return jsonify({"status": "error", "msg": "No file"}), 400

    # Save file
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        print("📄 Loading PDF...")
        docs = pdf_loader(path)

        print("✂️ Chunking...")
        chunks = chunk_documents(docs)

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

@app.route("/ask", methods=["POST"])
def ask():
    global rag_pipeline

    data = request.get_json()
    query = data["query"]

    if not rag_pipeline:
        return jsonify({"answer": "Please upload a document first."})

    answer, sources = rag_pipeline.run(query)

    return jsonify({
        "answer": answer
    })


# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)
