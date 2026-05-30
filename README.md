# 🕉 Sanskrit RAG System

A CPU-Optimized Hybrid Retrieval-Augmented Generation (RAG) Framework for Sanskrit Document Understanding and Context-Aware Question Answering.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)
![BM25](https://img.shields.io/badge/BM25-Keyword_Retrieval-orange)
![TinyLlama](https://img.shields.io/badge/TinyLlama-CPU_Inference-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-ff4b4b)

---

# 📌 Overview

The Sanskrit RAG System is a Retrieval-Augmented Generation (RAG) framework designed specifically for Sanskrit documents. The system combines semantic search, keyword retrieval, and lightweight Large Language Models to generate accurate, context-aware answers from Sanskrit knowledge sources.

Unlike traditional LLM applications, this system is optimized to run entirely on CPU hardware, making it lightweight, portable, and cost-efficient.

The architecture addresses key Sanskrit language challenges such as:

* Devanagari script processing
* Sandhi-based compound words
* Sanskrit-specific punctuation handling
* Domain-specific terminology retrieval
* Context preservation across shlokas and verses

---

# 🚀 Key Features

## 📄 Document Processing

* PDF Support
* TXT Support
* DOCX Support
* CSV Support
* PPTX Support
* Automatic Text Extraction
* Unicode Normalization

## 🔍 Hybrid Retrieval

* FAISS Semantic Search
* BM25 Keyword Search
* Reciprocal Rank Fusion (RRF)
* Context Ranking
* Top-K Retrieval

## 🧠 Sanskrit Processing

* Sanskrit Text Cleaning
* Devanagari Support
* Transliteration Pipeline
* Sandhi-Aware Chunking
* Shloka Preservation

## 🤖 Generation Layer

* TinyLlama Integration
* Quantized GGUF Models
* CPU-Only Inference
* Low Memory Footprint
* Context-Aware Response Generation

## 🌐 User Interface

* Streamlit Dashboard
* Interactive Querying
* Retrieval Visualization
* Response Monitoring

---

# 🏗 System Architecture

```text
                     Sanskrit RAG Pipeline

┌────────────────────────────────────────────────────┐
│                RAW DOCUMENTS                        │
│        PDF | TXT | DOCX | CSV | PPTX               │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│              DOCUMENT LOADER                       │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│       TEXT CLEANING & NORMALIZATION                │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│             TRANSLITERATION LAYER                  │
│          (Devanagari → IAST)                       │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│                CHUNK GENERATION                    │
└────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼

┌────────────────┐         ┌────────────────┐
│     FAISS      │         │      BM25      │
│ Dense Retrieval│         │Sparse Retrieval│
└────────────────┘         └────────────────┘
          │                           │
          └─────────────┬─────────────┘
                        ▼

┌────────────────────────────────────────────────────┐
│     HYBRID RETRIEVAL (RRF FUSION)                  │
└────────────────────────────────────────────────────┘
                        │
                        ▼

┌────────────────────────────────────────────────────┐
│              PROMPT ENGINEERING                    │
└────────────────────────────────────────────────────┘
                        │
                        ▼

┌────────────────────────────────────────────────────┐
│             TINYLLAMA (CPU INFERENCE)              │
└────────────────────────────────────────────────────┘
                        │
                        ▼

┌────────────────────────────────────────────────────┐
│               GENERATED ANSWER                     │
└────────────────────────────────────────────────────┘
```

---

# 🔄 Working Flow

### Step 1: Document Ingestion

Documents are uploaded into the system.

Supported formats:

* PDF
* TXT
* DOCX
* CSV
* PPTX

### Step 2: Text Processing

The extracted Sanskrit text undergoes:

* Unicode normalization
* Character cleaning
* Metadata removal
* Sanskrit punctuation preservation

### Step 3: Transliteration

The text is transformed into alternate representations:

* Devanagari
* IAST
* Optional SLP1

This improves retrieval accuracy.

### Step 4: Chunking

Documents are divided into meaningful chunks while preserving:

* Shlokas
* Danda boundaries (|)
* Double Danda boundaries (||)

### Step 5: Index Creation

Two retrieval systems are built:

#### Dense Retrieval

* Sentence Transformer Embeddings
* FAISS Vector Index

#### Sparse Retrieval

* BM25 Index
* Exact Keyword Matching

### Step 6: Hybrid Retrieval

FAISS and BM25 results are merged using Reciprocal Rank Fusion (RRF).

### Step 7: Prompt Construction

Retrieved context is combined with the user query.

### Step 8: Response Generation

TinyLlama generates a context-aware answer using the retrieved Sanskrit knowledge.

---

# 📂 Project Structure

```text
Sanskrit-RAG-System/

├── artifacts/
│   ├── faiss_index/
│   ├── bm25_index/
│   ├── logs/
│   ├── metrics/
│   └── responses/

├── code/
│   ├── app.py
│   ├── requirements.txt
│   └── src/
│       ├── pipeline/
│       ├── processors/
│       ├── retrieval/
│       └── generator/

├── data/
│   └── raw/

├── assets/
│   └── workflow_diagram.png

└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/prashantdarshanwar/RAG_SANSKRIT_PRASHANTDARSHANWAR

cd Sanskrit-RAG-System
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r code/requirements.txt
```

---

# 📥 Add Sanskrit Documents

Place files inside:

```text
data/raw/
```

Supported formats:

* PDF
* TXT
* DOCX
* CSV
* PPTX

---

# 🔨 Build Retrieval Index

```bash
python -m code.src.pipeline.rag_pipeline
```

Generated artifacts:

```text
artifacts/faiss_index/
artifacts/bm25_index/
artifacts/logs/
artifacts/metrics/
artifacts/responses/
```

---

# 🌐 Run Streamlit Application

```bash
streamlit run code/app.py
```

Open:

```text
http://localhost:8501
```

---

# 📊 Retrieval Formula

The system combines FAISS and BM25 rankings using Reciprocal Rank Fusion (RRF):

Score(d) = 1/(k + Rank_FAISS) + 1/(k + Rank_BM25)

Where:

* d = Retrieved Chunk
* k = 60

This balances semantic relevance and keyword relevance.

---

# 📈 Performance Characteristics

| Component             | Purpose             |
| --------------------- | ------------------- |
| FAISS                 | Semantic Retrieval  |
| BM25                  | Keyword Retrieval   |
| TinyLlama             | Response Generation |
| Streamlit             | User Interface      |
| Sentence Transformers | Embedding Creation  |

---

# 🛠 Tech Stack

* Python
* Streamlit
* FAISS
* Rank-BM25
* TinyLlama
* Sentence Transformers
* NumPy
* Pandas
* llama.cpp

---

# 🔮 Future Enhancements

* OCR Support for Sanskrit Manuscripts
* Docker Deployment
* FastAPI Backend
* Agentic RAG Workflows
* Knowledge Graph Integration
* Re-ranking Models
* Multi-Language Support

---

# 📜 License

This project is intended for educational, research, and Sanskrit language understanding purposes.

---

# 👨‍💻 Author

**Prashant Ashok Darshanwar**

AI/ML Engineer | GenAI Developer | RAG Systems Builder


