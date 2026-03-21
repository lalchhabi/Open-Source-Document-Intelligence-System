# 📄 Document Intelligence System (RAG-Based)

An end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to upload documents (PDFs) and ask intelligent, context-aware questions using a local LLM.

---

## 🚀 Features

- 📂 Upload PDF documents
- ✂️ Smart text chunking with overlap
- 🧠 Semantic search using embeddings (FAISS)
- 🤖 LLM-powered answer generation
- 💬 Context-aware multi-turn conversation
- 📊 Source-based answers (with page references)
- 🌐 Clean Flask-based web UI
- ⚡ Real-time processing with progress bar

---

## 🧠 System Architecture

```

User Query
↓
Retriever (FAISS Vector Search)
↓
Top-K Relevant Chunks
↓
Prompt Builder (Context + Chat History)
↓
LLM (Gemma 2B - HuggingFace)
↓
Final Answer

```

---

## 🏗️ Project Structure

```

Document-Intelligence/
│
├── app.py                  # Flask application
│
├── ingestion/
│   └── loader.py          # PDF loading & extraction
│
├── chunking/
│   └── chunker.py         # Text chunking logic
│
├── embeddings/
│   ├── embedder.py        # Embedding model
│   └── vector_store.py    # FAISS vector DB
│
├── retriever/
│   └── retriever.py       # Semantic retrieval
│
├── pipelines/
│   └── rag_pipelines.py   # RAG pipeline orchestration
│
├── llm/
│   ├── hf_model.py        # Model loading
│   ├── generator.py       # Answer generation
│   └── prompt_build.py    # Prompt engineering
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Frontend logic
│
└── uploads/               # Uploaded PDFs

```

---

## ⚙️ Tech Stack

- **Backend:** Flask  
- **LLM:** HuggingFace (`google/gemma-2b-it`)  
- **Embeddings:** SentenceTransformers (`bge-small-en`)  
- **Vector DB:** FAISS  
- **Frontend:** HTML, CSS, JavaScript  
- **PDF Processing:** PyMuPDF (fitz)  

---

## 🔄 How It Works

### 1. Document Ingestion
- Upload PDF
- Extract text from pages
- Clean and preprocess text

### 2. Chunking
- Split text into overlapping chunks
- Improves retrieval accuracy

### 3. Embedding + Storage
- Convert chunks → vector embeddings
- Store in FAISS index

### 4. Retrieval
- Convert query → embedding
- Retrieve top-K relevant chunks

### 5. Prompt Construction
- Combine:
  - Retrieved context
  - Chat history
  - User query

### 6. Answer Generation
- Pass prompt to LLM
- Generate contextual answer

---

## 💬 Context-Aware Chat

The system supports multi-turn conversations by maintaining:

```

chat_history = [
{"user": "...", "assistant": "..."}
]

````

Only the last few interactions are used to avoid token overflow.

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/lalchhabi/Open-Source-Document-Intelligence-System.git
cd document-intelligence
````

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Application

```bash
python app.py
```

---

### 4. Open in Browser

```
http://127.0.0.1:5000
```

---

## 📸 Demo Workflow

1. Upload a PDF
2. Wait for processing
3. Ask questions
4. Get AI-generated answers with sources

---

## ⚠️ Limitations

* Uses global memory (not multi-user safe)
* Model loading is heavy (can be optimized)
* Limited to PDF input
* No authentication system

---

## 🚀 Future Improvements

* ✅ Add user sessions (multi-user support)
* ✅ Use faster embedding models
* ✅ Add reranking (improve accuracy)
* ✅ Stream responses (ChatGPT-like UI)
* ✅ Deploy on cloud (AWS / GCP)
* ✅ Add support for DOCX, TXT
* ✅ Integrate evaluation metrics

---

## 📊 Evaluation (Planned)

* Retrieval accuracy (Top-K relevance)
* Answer correctness
* Latency (response time)
* Hallucination rate

---

## 🧠 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases (FAISS)
* Prompt Engineering
* Context-Aware Chat Systems

---

## 🙌 Author

**Chhabi Lal Tamang**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub and share it!
