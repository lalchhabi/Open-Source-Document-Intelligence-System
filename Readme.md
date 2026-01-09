# 📄 Open-Source Document Intelligence System

### Reliable RAG with Open-Source LLMs & Evaluation

## 🔍 Overview

This project implements an **open-source Document Intelligence System** that enables accurate, grounded question answering over unstructured documents using **Retrieval-Augmented Generation (RAG)**.

Unlike prompt-only chatbots, this system focuses on:

* **Reducing hallucinations**
* **Grounding answers in source documents**
* **Quantitatively evaluating LLM outputs**

The goal is to build a **reliable, transparent, and extensible RAG pipeline** suitable for real-world document-heavy applications.

---

## 🚩 Problem Statement

Organizations rely on large volumes of documents such as:

* Policies
* Contracts
* Reports
* Manuals
* Research papers

Traditional keyword search fails to capture semantic meaning, while LLMs alone tend to hallucinate when knowledge is missing.

**This project addresses the gap by combining retrieval + generation + evaluation** to ensure answers are:

* Relevant
* Faithful to source documents
* Measurable in quality

---

## 🧠 Key Features

* 📄 Multi-document ingestion (PDF/Text)
* ✂️ Configurable document chunking
* 🧠 Semantic search using vector embeddings
* 🔗 Retrieval-Augmented Generation (RAG)
* 📊 LLM output evaluation (faithfulness, relevance)
* 📚 Source citations for transparency
* 🧪 Error analysis & iterative improvement
* 🖥️ Interactive UI (Streamlit)
* ⚙️ API-first design (FastAPI)

---

## 🏗️ System Architecture

```
Documents
   ↓
Text Cleaning & Parsing
   ↓
Chunking & Metadata
   ↓
Embedding Generation
   ↓
Vector Database (FAISS)
   ↓
Custom Retriever
   ↓
Prompt Builder
   ↓
LLM (HuggingFace)
   ↓
Answer Generation
   ↓
Evaluation (RAGAS)
   ↓
UI / API
```

---

## 🛠️ Tech Stack

### Core

* **Python 3.10+**

### Document Processing

* PyMuPDF / pdfplumber
* LangChain text splitters

### Embeddings

* Sentence-Transformers

  * `all-MiniLM-L6-v2`
  * `bge-small-en`

### Vector Database

* FAISS

### LLMs (Open-Source)

* Mistral 7B Instruct / Qwen2.5 / Phi-3
* HuggingFace Inference API / Ollama

### RAG Framework
* Custom RAG pipeline (framework-light)
* Selective use of LangChain utilities (text splitting)


### Evaluation

* RAGAS
* Custom metrics

### Backend & UI

* FastAPI
* Streamlit

---

## 📂 Project Structure

```
document-intelligence-rag/
│
├── data/
│   ├── raw_docs/
│   └── processed_docs/
│
├── ingestion/
│   ├── loader.py
│   └── cleaner.py
│
├── chunking/
│   └── chunker.py
│
├── embeddings/
│   └── embedder.py
│
├── vectorstore/
│   └── faiss_index.py
│
├── retriever/
│   └── retriever.py
│
├── rag/
│   └── rag_pipeline.py
│
├── evaluation/
│   ├── ragas_eval.py
│   └── metrics.py
│
├── api/
│   └── app.py
│
├── ui/
│   └── streamlit_app.py
│
├── experiments/
│   └── logs/
│
├── README.md
└── requirements.txt
```

---

## 🚀 How It Works

1. **Ingest Documents**
   Load and clean PDFs or text files.

2. **Chunk Documents**
   Split documents into semantically meaningful chunks with overlap.

3. **Generate Embeddings**
   Convert chunks into vector representations.

4. **Store in Vector Database**
   Store embeddings in FAISS for fast similarity search.

5. **Retrieve Relevant Context**
   Retrieve top-k chunks for a user query.

6. **Generate Grounded Answers**
   LLM answers strictly using retrieved context.

7. **Evaluate Outputs**
   Measure faithfulness, relevance, and context alignment.

---

## 📊 Evaluation Strategy

The system uses **RAGAS** to evaluate LLM performance:

* **Faithfulness** – Is the answer supported by the retrieved context?
* **Answer Relevance** – Does the answer address the question?
* **Context Relevance** – Are retrieved documents useful?

Evaluation metrics guide **system improvements**, not just demos.

---

## 🎯 Use Cases

* Internal knowledge assistants
* Policy & compliance search
* Research document analysis
* Contract & legal document QA
* Technical documentation bots

---

## 🧪 Current Status

* [x] Document ingestion
* [x] Chunking & embedding
* [x] Vector search
* [ ] End-to-end RAG pipeline
* [ ] Evaluation pipeline
* [ ] UI & API
* [ ] Performance benchmarking

---

## 📌 Roadmap

* Improve chunking strategies
* Compare embedding models
* Add multi-query retrieval
* Implement answer citation highlighting
* Add experiment tracking

---

## 🤝 Contributions

This project is open to improvements, discussions, and experimentation.
Feel free to fork or open issues.

---

## 📜 License

MIT License

---

## 🧠 Author

**Chhabi Lal Tamang**
Machine Learning Engineer | LLM & RAG Systems
GitHub: [https://github.com/lalchhabi]

---
