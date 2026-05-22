# Import libraries 
from flask import Flask, request, Response, render_template, jsonify
import os
from llm.prompt_build import generate_chat_title
from config.state import app_state
import uuid
from utils.title_generator import finalize_title

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
chat_sessions = {}

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

    global chat_sessions, app_state

    data = request.get_json(silent=True) or {}
    old_id = data.get("session_id")
    llm = init_llm()

    # finalize old chat title
    if old_id in chat_sessions:
        finalize_title(chat_sessions[old_id], llm)

    new_id = str(uuid.uuid4())

    chat_sessions[new_id] = {
        "title": "New Chat",
        "messages": [],
        "mode": app_state["mode"],
        "title_generated": False
    }
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
    
    # Get session 
    session = chat_sessions.get(session_id)
    if not session:
        return jsonify({
            'error':'Invalid session'
        }), 400
    
    chat_history = session['messages']
    
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
        chat_history.append({
            "role":"user",
            "content":query
        })

        chat_history.append({
            "role":'assistant',
            "content": full_answer
        })
        # keep last 5 exchanges(10 messages)
        session['messages'] = chat_history[-10:]

        # Title generation logic

        llm = init_llm()
        # if chat is still short -> wait
        if len(session['messages']) < 6:
            pass
        
        # if enough messages -> generate title once
        elif not session.get("title_generated", False):
            
            title = generate_chat_title(llm, session['messages'])
            session['title'] = title
            session['title_generated'] = True

    return Response(
        generate(), 
        mimetype="text/plain")

# Get sidebar chat list
@app.route("/sessions", methods = ["GET"])
def get_sessions():
    return jsonify([
        {
            "session_id": sid,
            "title": data['title']
        }
        for sid, data in chat_sessions.items()
        if len(data['messages']) > 0
    ])

# Load specific chat
@app.route("/session/<session_id>", methods = ['GET'])
def load_session(session_id):
    session = chat_sessions.get(session_id)
    if not session:
        return jsonify({
            "error": "not found"
        }), 404
    return jsonify(session)

# if user leaves early
@app.route("/close-session", methods = ['POST'])
def close_session():
    session_id = request.json.get('session_id')

    if session_id in chat_sessions:
        llm = init_llm()
        finalize_title(chat_sessions[session_id], llm)

    return jsonify({'status': "ok"})


# Run App
if __name__ == "__main__":
    app.run(debug = True, use_reloader = False)