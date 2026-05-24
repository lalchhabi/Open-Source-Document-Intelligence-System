# 📄 Open-Source Document Intelligence System

An AI-powered hybrid chat and Retrieval-Augmented Generation (RAG) system built with Flask, LangChain, FAISS, and Large Language Models.

This project enables users to:

* Upload PDF documents
* Ask document-specific questions using RAG
* Use normal conversational AI chat without documents
* Maintain multi-session chat history
* Persist chats using SQLite
* Stream AI responses in real time
* Switch intelligently between normal chat mode and RAG mode

---

# 🚀 Features

## ✅ Hybrid Chat System

The application supports two intelligent modes:

### 1. Normal Chat Mode

* Direct LLM interaction
* General-purpose AI assistant
* No document retrieval required

### 2. RAG Mode

* Upload PDF documents
* Semantic retrieval from uploaded files
* Context-aware answers grounded in documents
* Hybrid dense + sparse retrieval pipeline

---

# 🧠 Core Implementations

## 📌 Retrieval-Augmented Generation (RAG)

The system uses a complete RAG pipeline:

1. PDF loading
2. Text cleaning
3. Chunking
4. Embedding generation
5. Vector storage
6. Hybrid retrieval
7. Reranking
8. Prompt construction
9. LLM response generation
10. Streaming response delivery

---

# ⚡ Current Functionalities

## ✅ PDF Upload System

* Upload PDF documents from frontend
* Automatic ingestion pipeline
* Document processing status updates

## ✅ Intelligent Chunking

Uses recursive text splitting for better semantic retrieval.

## ✅ Embedding Pipeline

Converts document chunks into vector embeddings for semantic search.

## ✅ FAISS Vector Database

Stores dense embeddings for fast similarity retrieval.

## ✅ Hybrid Retriever

Combines:

* Dense vector retrieval
* Sparse keyword retrieval

This improves factual accuracy and retrieval quality.

## ✅ Reranking Layer

Retrieved chunks are scored again using a Cross-Encoder
reranker to improve contextual relevance and reduce noisy retrievals.

## ✅ Streaming Responses

Responses are streamed token-by-token using Flask streaming.

## ✅ Context-Aware Conversations

Maintains short-term conversation memory during sessions.

## ✅ Multi-Session Chat System

* Create multiple chats
* Sidebar chat history
* Session switching
* Persistent chat storage

## ✅ SQLite Persistence

Stores:

* Chat sessions
* Chat titles
* User messages
* Assistant messages

## ✅ Dynamic Chat Title Generation

Chat titles are generated intelligently based on conversation context.

## ✅ Delete Chat Sessions

Users can delete individual chat sessions directly from sidebar.

## ✅ Document Remove System

Switch back from RAG mode to normal chat mode dynamically.

---

# 🏗️ System Architecture

```text
                ┌────────────────────┐
                │   User Interface   │
                │ HTML / CSS / JS    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    Flask Backend   │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                                    ▼
┌────────────────┐                 ┌────────────────┐
│   Normal Chat  │                 │    RAG Mode    │
│      LLM       │                 │ PDF + Retriever│
└────────────────┘                 └───────┬────────┘
                                           │
                              ┌────────────┴───────────┐
                              ▼                        ▼
                    ┌──────────────────┐ ┌──────────────────┐
                    │ Dense Retrieval  │ │Sparse Retrieval  │
                    └─────────┬────────┘ └─────────┬────────┘                                     
                              └────────────┬───────────┘
                                           ▼
                                 ┌──────────────────┐
                                 │    Reranker      │
                                 └─────────┬────────┘
                                           ▼
                                 ┌──────────────────┐
                                 │       LLM        │
                                 └─────────┬────────┘
                                           ▼
                                 ┌──────────────────┐
                                 │ Streamed Response│
                                 └──────────────────┘
```

---

# 🛠️ Tech Stack

## Backend

* Python
* Flask
* LangChain
* SQLite

## AI / NLP

* HuggingFace Transformers
* SentenceTransformers
* Retrieval-Augmented Generation (RAG)
* Hybrid Search
* Cross Encoder Reranking

## Vector Database

* FAISS

## Frontend

* HTML
* CSS
* Vanilla JavaScript

## LLM Integration

* HuggingFace Inference
* Groq API (optional support)

---

# 📂 Project Structure

```text
Document_Intelligence_System/
│
├── app.py
├── main.py
├── requirements.txt
├── .env
├── .gitignore
├── Readme.md
│
├── database/
│   ├── chatbot.db
│   └── db.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│
├── uploads/
│
├── ingestion/
│   └── loader.py
│
├── chunking/
│   └── chunker.py
│
├── embeddings/
│   └── embedder.py
│   └── vector_store.py
│
├── retriever/
│   └── retriever.py
│
├── reranker/
│   └── reranker.py
│
├── llm/
│   ├── hf_model.py
│   └── prompt_build.py
│
├── pipelines/
│   └── rag_pipeline.py
│
└── utils/
    └── title_generator.py
    └── text_cleaner.py
```

---

# 🔄 How the System Works

## Step 1: Upload Document

User uploads a PDF document.

## Step 2: PDF Loading

The PDF is parsed and converted into raw text.

## Step 3: Chunking

Large text is split into overlapping semantic chunks.

## Step 4: Embedding Generation

Chunks are converted into vector embeddings.

## Step 5: Vector Storage

Embeddings are stored inside FAISS.

## Step 6: Hybrid Retrieval

System retrieves relevant chunks using:

* Dense semantic search
* Sparse keyword search

## Step 7: Cross-Encoder Reranking

Retrieved chunks are scored again using a Cross-Encoder
reranker to improve contextual relevance and reduce noisy retrievals.

## Step 8: Prompt Construction

Context + user query + chat history are combined.

## Step 9: LLM Generation

The LLM generates grounded responses.

## Step 10: Streaming Response

Response is streamed token-by-token to frontend.

---

# 🧠 Context-Aware Chat System

The chatbot supports conversational memory during sessions.

Features:

* Maintains previous messages
* Supports follow-up questions
* Context-aware RAG responses
* Multi-session persistence

Example:

```text
User: Summarize chapter 1
User: Explain the second point more deeply
```

The system remembers earlier context.

---

# 💾 Database Design

## chat_sessions Table

Stores:

* session id
* title
* creation time

## messages Table

Stores:

* session id
* role (user/assistant)
* message content
* timestamps

---

# ▶️ How to Run the Project

## 1. Clone Repository

```bash
git clone <repository_url>
cd Document_Intelligence_System
```

## 2. Create Virtual Environment

```bash
python -m venv rag_doc
source rag_doc/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
LANGCHAIN_API_KEY=your_key
```

## 5. Run Application

```bash
python app.py
```

## 6. Open Browser

```text
http://127.0.0.1:5000
```

---

# ⚠️ Current Limitations

* Currently optimized mainly for PDF files
* SQLite is suitable for small-to-medium scale projects only
* Limited long-term memory
* Single-user local deployment focus
* No authentication system yet
* No cloud deployment yet
* Reranking can increase latency slightly

---

# 🚀 Planned Future Improvements

## 🔹 Authentication System

* User login/signup
* Personalized chat history

## 🔹 Cloud Deployment

* Docker
* AWS/GCP/Azure deployment

## 🔹 Advanced Vector Databases

* Pinecone
* Weaviate
* ChromaDB

## 🔹 Multi-Document Retrieval

Query across multiple uploaded documents.

## 🔹 Long-Term Memory

Persistent semantic memory system.

## 🔹 Advanced Agentic AI Workflow

* Tool calling
* Multi-agent orchestration
* LangGraph workflows

## 🔹 OCR Support

Support scanned PDFs and images.

## 🔹 Better UI/UX

* Typing animation
* Active session highlighting
* Dark/light themes
* Mobile responsiveness

## 🔹 Production Monitoring

* LangSmith tracing
* Observability dashboards
* Error monitoring

---

# 📊 Evaluation (Planned)

Future evaluation metrics:

* Retrieval accuracy
* Context relevance
* Hallucination reduction
* Latency benchmarking
* RAG response grounding
* User experience testing

---

# 📚 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Hybrid Retrieval
* Dense Retrieval
* Sparse Retrieval
* Vector Embeddings
* Similarity Search
* Reranking
* Context-Aware Chat
* Streaming Responses
* Session Persistence
* Conversational AI
* LLM Prompt Engineering

---

# 👨‍💻 Author

## Chhabi Lal Tamang

AI Engineer | NLP | Computer Vision | LLM Systems

Focused on:

* Agentic AI
* Retrieval-Augmented Generation
* LLM Applications
* Deep Learning Systems
* AI Product Engineering

---

# ⭐ Project Vision

The goal of this project is to build a production-style open-source AI document assistant capable of:

* intelligent retrieval
* conversational reasoning
* scalable memory systems
* real-world enterprise workflows

while learning modern LLM engineering practices.

---

# 📜 License

This project is open-source and available for educational and research purposes.
