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

## 🚀 Features

### 🤖 Dual Interaction Modes
* **Normal Chat Mode:** For general-purpose conversations using an LLM.
* **RAG Mode:** For document-grounded question answering after PDF upload.
* **Seamless Switching:** Easily switch between chat mode and document mode without restarting the application.

### 📄 Intelligent Document Processing
* **PDF Ingestion:** Complete with automatic parsing.
* **Recursive Document Chunking:** Optimized for semantic retrieval.
* **Dense Vector Embedding Generation:** For capturing deep contextual meaning.
* **FAISS Vector Indexing:** Ensures efficient and fast similarity search.

### 🔍 Hybrid Retrieval Pipeline
* **Dense Semantic Retrieval:** Powered by vector embeddings.
* **Sparse Keyword Retrieval:** Utilizes BM25 for precise keyword matching.
* **Cross-Encoder Reranking:** Cross-Encoder reranking with BAAI/bge-reranker-base.

### 💬 Context-Aware Conversations
* **Multi-Session Chat Support:** Keep track of different topics seamlessly.
* **Conversation History:** Maintained continuously for natural follow-up questions.
* **Automatic Chat Title Generation:** Dynamically named based on your conversation context.
* **Persistent Chat History:** Safely backed up and stored using SQLite.

### ⚡ Streaming AI Responses
* **Token-by-Token Streaming:** Real-time response streaming for an improved, snappy user experience.
* **Universal Support:** Works flawlessly in both Normal Chat and RAG modes.

### 📊 Observability & Evaluation
* **Pipeline Tracing:** End-to-end tracing and visibility powered by LangSmith.
* **Built-in RAGAS Framework:** Evaluation metrics for measuring precise retrieval and generation quality.

### 🐳 Deployment Ready
* **Dockerized Application:** Containerized for easy environment management.
* **Docker Compose Support:** Streamlined setup for quick local development.
* **Render Deployment:** Tested and successfully deployed live on Render.

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

## Frontend
- HTML
- CSS
- Vanilla JavaScript

## Backend
- Python
- Flask
- LangChain
- SQLite

## AI & LLM
- Meta-Llama-3-8B-Instruct (Response Generation)
- BAAI/bge-small-en (Embeddings)
- BAAI/bge-reranker-base (Cross-Encoder Reranker)

## RAG Pipeline
- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval (Dense + BM25)
- Prompt Engineering
- Streaming Responses

## Vector Database
- FAISS

## Evaluation
- RAGAS
- Groq API (LLM Evaluator)

## Observability
- LangSmith

## Deployment
- Docker
- Docker Compose
- Render

---

# 📂 Project Structure

```text
Document_Intelligence_System/
│
├── app.py
├── main.py
├── requirements.txt
├── requirements_ml.txt
├── requirements_dev.txt
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

### 1. Start a Chat Session
A new chat session is created and stored in SQLite. The system maintains conversation history and generates chat titles automatically.

### 2. Choose Interaction Mode
The application supports two modes:

- **Normal Chat Mode:** Queries are sent directly to the LLM.
- **RAG Mode:** Users upload a PDF, and responses are grounded in the document.

### 3. Document Processing (RAG Mode)
When a PDF is uploaded, the system:

- Extracts text from the document
- Cleans and preprocesses the content
- Splits the text into semantic chunks

### 4. Embedding & Indexing
Each chunk is converted into dense vector embeddings and indexed in FAISS for efficient semantic retrieval.

### 5. Hybrid Retrieval
For every user query, the system retrieves relevant document chunks using:

- Dense semantic search (FAISS)
- Sparse keyword search (BM25)

### 6. Cross-Encoder Reranking
Retrieved chunks are reranked using a Cross-Encoder model to improve context relevance before generation.

### 7. Prompt Construction
The final prompt combines:

- Conversation history
- Retrieved document context
- User query

This enables context-aware and grounded responses.

### 8. Response Generation
The Meta-Llama-3-8B-Instruct model generates the final response using the constructed prompt.

### 9. Streaming Response
Responses are streamed token-by-token to provide a more responsive chat experience.

### 10. Session Persistence
User messages, assistant responses, and generated chat titles are stored in SQLite, enabling multi-session conversations.

### 11. Observability
LangSmith traces the end-to-end RAG pipeline for debugging, monitoring, and performance analysis.
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

# 🧠 AI Components

The system combines multiple specialized AI models to build an end-to-end Retrieval-Augmented Generation (RAG) pipeline.

| Component | Model | Purpose |
|-----------|-------|---------|
| Response Generation | `meta-llama/Meta-Llama-3-8B-Instruct` | Generates conversational responses using retrieved document context or general chat mode. |
| Embedding Model | `BAAI/bge-small-en` | Converts document chunks into dense vector embeddings for semantic retrieval. |
| Sparse Retrieval | `BM25` | Performs keyword-based retrieval to complement dense vector search. |
| Reranker | `BAAI/bge-reranker-base` | Re-ranks retrieved documents using cross-encoder scoring before passing context to the LLM. |
| Vector Store | `FAISS` | Stores embeddings and performs efficient similarity search. |
| Evaluation LLM | `qwen/qwen3-32b` (Groq) | Used only for automated RAGAS evaluation metrics. |

# 📊 Evaluation

This project includes an automated evaluation pipeline using **RAGAS (Retrieval-Augmented Generation Assessment)** to measure the quality of the production RAG system.

The evaluation focuses on:

- Answer correctness
- Retrieval quality
- Context relevance
- Faithfulness to retrieved documents

---

## Evaluation Workflow

For each evaluation question, the system:

1. Loads the reference document
2. Builds the same production RAG pipeline
3. Generates an answer using the LLM
4. Collects the retrieved contexts and ground truth
5. Evaluates the response using RAGAS metrics

---

## Evaluation Dataset

The evaluation dataset consists of question and ground-truth answer pairs.

Example:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "ground_truth": "Retrieval-Augmented Generation combines retrieval and generation using external knowledge."
}
```

---

## Evaluation Components

| Component | Model / Tool | Purpose |
|-----------|--------------|---------|
| **Evaluation Framework** | RAGAS | Measures retrieval and generation quality. |
| **Evaluator LLM** | `qwen/qwen3-32b` (Groq) | Evaluates faithfulness and answer relevancy. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Computes embedding-based similarity metrics. |
| **Answer Generation** | Production RAG Pipeline | Generates responses to be evaluated. |

---

## Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Measures whether the generated answer is supported by the retrieved context. |
| **Answer Relevancy** | Measures how well the response answers the user's question. |
| **Context Precision** | Evaluates whether retrieved chunks are relevant to the query. |
| **Context Recall** | Measures whether all required information was successfully retrieved. |

---

## Example Results

| Metric | Score |
|---------|------:|
| Faithfulness | **1.00** |
| Answer Relevancy | **0.93** |
| Context Precision | **0.58** |
| Context Recall | **1.00** |

---

## Output

Evaluation results are automatically saved to:

```text
evaluation/results.json
```

Example:

```json
{
  "faithfulness": 1.0,
  "answer_relevancy": 0.93,
  "context_precision": 0.58,
  "context_recall": 1.0
}
```

---

# ▶️ How to Run the Project

## 1. Clone Repository

```bash
git clone <https://github.com/lalchhabi/Open-Source-Document-Intelligence-System/tree/feat/deploy-app>
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
pip install -r requirements_ml.txt
```

## 4. Configure Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
LANGCHAIN_API_KEY=your_key
COHERE_API_KEY=your_cohere_api_key
```

## 5. Run Application

```bash
python app.py
```

## 6. Open Browser

```text
http://127.0.0.1:5000
```

# 🐳 Docker Setup

The project can be run in an isolated container using Docker or Docker Compose.

### Build the Docker Image

```bash
docker build -t document-intelligence-system .
```

### Run the Container

```bash
docker run -p 5000:5000 \
  --env-file .env \
  document-intelligence-system
```

### Run with Docker Compose

```bash
docker compose up
```

The application will be available at:

```text
http://localhost:5000
```

---

# ⚙️ CI/CD Pipeline

This project includes a GitHub Actions workflow for continuous integration and is deployed using Render.

### CI Features

- Automatic dependency installation
- Docker image build verification
- Application startup validation
- Build checks on every push and pull request

### Deployment

- Dockerized application
- Automatic deployment from GitHub using Render
- Containerized production environment

---

# ⚠️ Current Limitations

- Optimized primarily for PDF documents.
- SQLite is intended for development and small-scale deployments.
- Uploaded documents are processed per session and are not persisted across deployments.
- Large PDF files may exceed the memory limits of free cloud hosting services.
- No user authentication or authorization system.
- Cross-Encoder reranking improves retrieval quality but increases response latency.

---

# 🚀 Future Improvements

- **User Authentication**
  - User registration and login
  - Personalized chat history

- **Persistent Vector Database**
  - Replace in-memory FAISS with persistent vector databases such as Pinecone, Weaviate, or ChromaDB.

- **Multi-Document Retrieval**
  - Query across multiple uploaded documents simultaneously.

- **Long-Term Memory**
  - Persistent conversational memory across sessions.

- **OCR Support**
  - Support scanned PDFs and image-based documents.

- **Advanced Agentic Workflows**
  - Tool calling
  - Multi-agent orchestration
  - LangGraph-based workflows

- **Enhanced UI/UX**
  - Typing indicators
  - Mobile responsiveness
  - Dark/Light theme
  - Better document management
  - Upload progress indicators

- **Production Infrastructure**
  - PostgreSQL database
  - Redis caching
  - Scalable cloud deployment (AWS/GCP/Azure)

## 👨‍💻 Author

**Chhabi Lal Tamang**

AI Engineer passionate about building production-ready AI applications with Large Language Models, Retrieval-Augmented Generation (RAG), Computer Vision, and Deep Learning.

- GitHub: https://github.com/lalchhabi
- LinkedIn: https://www.linkedin.com/in/chhabi-lal-tamang-6a6b71222/

---

# 📜 License

This project is open-source and available for educational and research purposes.
