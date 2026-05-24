# Import libraries 
from flask import Flask, request, Response, render_template, jsonify
import os
from llm.prompt_build import generate_chat_title
from config.state import app_state
import uuid
from utils.title_generator import finalize_title
from langsmith import traceable

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

# Import database file
from database.db import *

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global RAG pipeline (shared across requests)
rag_pipeline = None

# general chat model
llm = None

# Load general llm on start
def init_llm():
    global llm
    if llm is None:
        llm = load_llm()

    return llm

# MAIN PAGE ROUTE
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
    """Upload document and build RAG pipeline.

    After upload:
    - System switches to RAG mode explicitly
    - Chat history is reset
    - Pipeline becomes active for /ask endpoint
    """

    global rag_pipeline, chat_history, app_mode

    file = request.files.get("pdf")
    if not file:
        return jsonify({'status':'error', "msg":"No file uploaded"}),400
    
    # Save file
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

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
            vector_store=vector_store
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

        # Implement Switch Mode here
        app_state['mode'] = 'rag'
        app_state['rag_pipeline'] = rag_pipeline

        return jsonify({
            "status": "success",
            "mode": "rag",
            "msg": "Document Uploaded Successfully. RAG mode enabled."
        })
    
    except Exception as e:
        return jsonify({"status":'error', 'msg':str(e)}), 500
    
@app.route("/remove-doc", methods=["POST"])
def remove_doc():
    global rag_pipeline

    rag_pipeline = None

    app_state["mode"] = "chat"
    app_state["rag_pipeline"] = None

    return jsonify({"status": "switched to chat mode",
                    "msg":"Document removed. Normal chat mode enabled."})

# Route for new chat
@app.route("/new-chat", methods = ['POST'])
def new_chat():

    global app_state

    data = request.get_json(silent=True) or {}
    old_id = data.get("session_id")
    llm = init_llm()

    # finalize previous old chat title
    if old_id:
        finalize_title(old_id, llm)

    # Create new DB session
    new_id = create_session()

    return jsonify({
        "session_id": new_id,
        "title": "New Chat"
    })

# Chat Endpoint (Smart Routing)
@app.route("/ask", methods = ['POST'])
def ask():
    """
    Streaming endpoint for both:
    1. Normal Chat Mode (LLM only)
    2. RAG Mode (retrieval + LLM)

    Response is streamed token-by-token using Server-Sent Streaming.
    """

    global rag_pipeline, app_state

    data = request.get_json()
    query = data.get("query", "")
    session_id = data.get("session_id")

    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    # Validate session from database
    if not session_exists(session_id):
        return jsonify({
            'error':'Invalid session'
        }), 400
    
    chat_history = get_session_messages(session_id)
    
    @traceable(name = "ask_endpoint")
    def generate():
        """Streaming generator for Flask response.

        This function handles real-time token streaming for both:
        - RAG-based responses (with document retrieval)
        - Normal LLM chat responses

        """
        full_answer = ""

        try:
            # CASE 1: RAG MODE
            if (app_state['mode'] == 'rag' and app_state['rag_pipeline'] is not None):
                stream, sources = app_state['rag_pipeline'].stream(query=query,
                chat_history=chat_history
                )
                
                for chunk in stream:
                    token = chunk.content
                    full_answer += token
                    yield token

            # Case 2: Normal Chat Mode
            else:
                llm = init_llm()
                stream = llm.stream(query)

                for chunk in stream:
                    token = chunk.content
                    full_answer += token
                    yield token
            
        except Exception as e:
            yield f"\n[ERROR]: {str(e)}"

        # Save message to session
        save_message(
            session_id=session_id,
            role='user',
            content=query
        )

        save_message(
            session_id=session_id,
            role="assistant",
            content=full_answer
        )

        # Title generation logic
        messages = get_session_messages(session_id)

        # Only generate title if there is enough messages

        if len(messages) >= 4:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            current_title = row['title'] if row else None
            if not current_title or current_title == "New Chat":
                llm = init_llm()
                title = generate_chat_title(llm,messages)
                update_session_title(session_id, title)
    return Response(
        generate(), 
        mimetype="text/plain")

# Get side bar chat list
@app.route("/sessions", methods = ['GET'])
def sessions():
    """API to Load sessions
    """
    sessions = get_all_sessions()
    formatted = [
        {
            "id": s['id'],
            "title": s['title']
        }
        for s in sessions
    ]
    return jsonify(formatted)

# Load specific chat
@app.route("/session/<session_id>", methods = ['GET'])
def load_session(session_id):
    session = get_session(session_id)
    if not session:
        return jsonify({
            "error": "not found"
        }), 404
    
    messages = get_session_messages(session_id)
    return jsonify({
        "session": session,
        "messages": messages
    })

# delete specific chat
@app.route("/session/<session_id>", methods = ["DELETE"])

def delete_chat(session_id):

    delete_session(session_id)
    return jsonify({
        "status":'deleted',
        "session_id": session_id
    })

# if user leaves early
@app.route("/close-session", methods = ['POST'])
def close_session():
    session_id = request.json.get('session_id')

    if session_exists(session_id):
        llm = init_llm()

        finalize_title(
            session_id,
            llm
        )
    return jsonify({
        "status":"ok"
    })

# Run App
if __name__ == "__main__":
    init_db()
    app.run(debug = True, use_reloader = False)