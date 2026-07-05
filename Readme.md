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

## ✅ LangSmith observability and tracing support
End-to-end RAG pipeline monitoring and Streaming response tracing
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
                                 └─────────┬────────┘
                                           ▼
                                 ┌──────────────────┐
                                 │ LangSmith Tracing│
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

## Observability & Monitoring
* LangSmith

---

# 📂 Project Structure

```text
Document_Intelligence_System/
│
├── app.py
├── main.py
├── requirements.txt
├── requirements_ml.txt
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
├── evaluation/
│    └── evaluation.py
│    └── results.json
│    └── sample_question.json
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
    └── load_vector_store.py
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

## Step 11: Tracing and Monitoring

LangSmith traces and monitors the complete RAG workflow for observability and debugging.


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

## 🧠 Models & Frameworks Used

This system uses a combination of open-source LLMs and transformer-based models across embedding, retrieval, reranking, and generation stages.

### 🔹 Main LLM (Response Generation)
* **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Provider:** Hugging Face Inference API
* **Task Type:** Conversational LLM
* **Usage:** Final answer generation in RAG pipeline

### 🔹 Embedding Model
* **Model:** `BAAI/bge-small-en`
* **Purpose:** Semantic embedding for document chunking and FAISS retrieval
* **Output:** Dense vector representations for similarity search

### 🔹 Reranker Model
* **Model:** `BAAI/bge-reranker-base`
* **Type:** Cross-Encoder Transformer
* **Purpose:** Re-ranking retrieved chunks for better context relevance
* **Role:** Improves precision before LLM generation

### 🔹 Vector Database
* **FAISS (Facebook AI Similarity Search)**
  * Used for efficient similarity-based retrieval of document chunks.

### 🔹 Frameworks & Libraries
* **LangChain** → RAG pipeline orchestration
* **Hugging Face Transformers** → LLM + embeddings + reranker
* **Flask** → Backend API system
* **RAGAS** → Evaluation framework
* **LangSmith** → Tracing and observability

# 📊 Evaluation 
## Overview

This project includes a fully automated evaluation pipeline using RAGAS (Retrieval-Augmented Generation Assessment) to measure the performance of the RAG system.

The evaluation is designed to test:
* Answer correctness
* Retrieval quality
* Context relevance
* Faithfulness to source document

---

## Evaluation Pipeline Setup

The evaluation system uses the same RAG pipeline used in production, ensuring realistic benchmarking.

---

## Dataset Format

We use a structured JSON dataset:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "ground_truth": "Retrieval-Augmented Generation combines retrieval and generation using external knowledge."
}

```

---

## Evaluation Workflow

### For each question:

* Load PDF document (Resume / Knowledge base)
* Chunk documents using recursive text splitter
* Generate embeddings using SentenceTransformers
* Store vectors in FAISS
* Retrieve relevant chunks using Hybrid Retriever
* Generate answer using LLM

### Collect:

* User question
* Model response
* Retrieved contexts
* Ground truth
* Build evaluation dataset
* Run RAGAS metrics evaluation

---

## Models Used in Evaluation

### 🔹 LLM (Evaluator Model)

* **Provider:** Groq API
* **Model:** `qwen/qwen3-32b`
* **Temperature:** 0
* **Used for:**
* Faithfulness evaluation
* Answer relevancy scoring



### 🔹 Embedding Model (RAGAS Evaluation)

* **Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Provider:** HuggingFace
* **Used for:**
* Context similarity measurement
* Retrieval evaluation



### 🔹 RAG Pipeline LLM (Answer Generation)

* **Model:** Configurable HuggingFace / Groq model
* **Used for:** Generating system responses for evaluation

---

## Metrics Used (RAGAS)

1. **Faithfulness**
* Checks if the answer is supported by retrieved context.
* ✔ Detects hallucinations


2. **Answer Relevancy**
* Measures how relevant the answer is to the question.
* ✔ Ensures correct response focus


3. **Context Precision**
* Measures how relevant retrieved chunks are.
* ✔ Evaluates retrieval noise


4. **Context Recall**
* Measures whether all necessary information was retrieved.
* ✔ Ensures completeness of retrieval

---

## Example Results

| Metric | Score |
| --- | --- |
| **Faithfulness** | 1.00 |
| **Answer Relevancy** | 0.93 |
| **Context Precision** | 0.58 |
| **Context Recall** | 1.00 |

---

## Output Storage

Evaluation results are automatically saved as:
`evaluation/results.json`

**Stored Format:**

```json
{
  "faithfulness": 1.0,
  "answer_relevancy": 0.93,
  "context_precision": 0.58,
  "context_recall": 1.0
}

```

---

## Why We Use a Fixed Dataset

RAG systems are probabilistic, so evaluation requires:

* Same questions every run
* Known ground truth answers
* Stable benchmarking

This ensures reproducibility and fair comparison.

---

## Tools Used

* **RAGAS** → Evaluation framework
* **Groq API** → LLM evaluator
* **HuggingFace Transformers** → Embeddings + inference
* **FAISS** → Vector database
* **LangChain** → Pipeline orchestration

---

## Key Insight

This evaluation pipeline ensures:

* ✔ End-to-end RAG validation
* ✔ Retrieval + generation quality measurement
* ✔ Production-ready benchmarking system
* ✔ Automated result tracking

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

## 🐳 Docker Setup

This project supports containerized execution using Docker.

### Build Image
docker build -t rag-app .

### Run Container
docker run -p 5000:5000 rag-app

## ⚙️ CI/CD Pipeline

This project includes a GitHub Actions CI pipeline.

### Features:
- Automatic build validation
- Dependency installation check
- Docker build verification
- Code stability check on every push

📌 Note: CD (deployment) is kept as future enhancement due to infrastructure limitations.

---

# ⚠️ Current Limitations

* Currently optimized mainly for PDF files
* SQLite is suitable for small-to-medium scale projects only
* Limited long-term memory
* Single-user local deployment focus
* No authentication system yet
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
* Observability with LangSmith
* RAGAS for Evaluation

---

# 👨‍💻 Author

## Chhabi Lal Tamang

AI Engineer | LLM Systems | RAG | Computer Vision 

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
