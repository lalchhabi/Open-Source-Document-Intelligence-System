from flask import Flask, request, render_template

from ingestion.loader import pdf_loader
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from embeddings.vector_store import FAISSStore
from retriever.retriever import Retriever
from pipelines.rag_pipelines import RAGPipeline

import os

app = Flask(__name__)

# Global pipeline (simple approach)
rag_pipeline = None


@app.route("/", methods=["GET", "POST"])
def index():
    global rag_pipeline

    answer = None
    sources = []
    message = None  

    if request.method == "POST":

        # Upload PDF
        if "pdf" in request.files and request.files["pdf"].filename != "":
            try:
                file = request.files["pdf"]

                os.makedirs("uploads", exist_ok=True)
                path = os.path.join("uploads", file.filename)
                file.save(path)

                message = "✅ PDF uploaded and processing..."

                # Build pipeline
                docs = pdf_loader(path)
                chunks = chunk_documents(docs)

                texts = [c["text"] for c in chunks]
                metadata = [c["metadata"] for c in chunks]

                embedder = Embedder()
                embeddings = embedder.embed_texts(texts)

                store = FAISSStore(len(embeddings[0]))
                store.add(embeddings, texts, metadata)

                retriever = Retriever(
                    embedder=embedder,
                    vector_store=store
                )

                rag_pipeline = RAGPipeline(retriever)

                message = "✅ Document processed successfully!"

            except Exception as e:
                message = f"❌ Error: {str(e)}"

        # Ask question
        if "query" in request.form and rag_pipeline:
            query = request.form["query"]
            answer, sources = rag_pipeline.run(query)

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        message=message   
    )

if __name__ == "__main__":
    app.run(debug=True)
