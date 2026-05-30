# 🕉 Sanskrit RAG System

A CPU-optimized Retrieval-Augmented Generation (RAG) system designed for Sanskrit document understanding and question answering. The system combines semantic search (FAISS), keyword retrieval (BM25), and lightweight Large Language Models (TinyLlama) to generate context-aware responses from Sanskrit knowledge sources.

---

## 📌 Overview

This project implements an end-to-end RAG pipeline capable of:

* Processing Sanskrit documents from multiple formats
* Performing intelligent semantic retrieval
* Supporting exact keyword matching
* Generating contextual answers using a lightweight LLM
* Running completely on CPU without GPU requirements

The architecture is designed specifically to address challenges in Sanskrit text processing such as Devanagari script handling, Sandhi-based compound words, and domain-specific terminology.

---

## 🚀 Key Features

### Document Processing

* PDF Support
* TXT Support
* DOCX Support
* CSV Support
* PPTX Support

### Retrieval System

* FAISS Vector Search
* BM25 Keyword Search
* Hybrid Retrieval Strategy
* Reciprocal Rank Fusion (RRF)

### Language Processing

* Sanskrit Text Cleaning
* Unicode Normalization
* Transliteration Support
* Sandhi-Aware Chunking

### Generation Layer

* TinyLlama Inference
* CPU-Only Execution
* Quantized GGUF Models
* Low Memory Footprint

### User Interface

* Streamlit Dashboard
* Interactive Query Interface
* Retrieval Monitoring
* Response Tracking

---

# 🏗 System Architecture

```text
Raw Documents
(PDF/TXT/DOCX/CSV/PPTX)
          │
          ▼
Document Loader
          │
          ▼
Text Cleaning & Normalization
          │
          ▼
Transliteration Layer
          │
          ▼
Chunk Generation
          │
 ┌────────┴─────────┐
 ▼                  ▼
FAISS           BM25
(Dense)        (Sparse)
 │                │
 └──────┬─────────┘
        ▼
Hybrid Retriever
(RRF Fusion)
        │
        ▼
Prompt Builder
        │
        ▼
TinyLlama
        │
        ▼
Generated Answer
```

---

## 🔍 Retrieval Pipeline

### Dense Retrieval (FAISS)

Text chunks are converted into embeddings using Sentence Transformers and stored inside a FAISS vector index.

Benefits:

* Semantic similarity matching
* Context-aware retrieval
* Fast vector search

### Sparse Retrieval (BM25)

BM25 performs lexical matching to capture exact Sanskrit terminology and important keywords.

Benefits:

* Precise keyword matching
* Improved recall
* Better handling of rare terms

### Hybrid Retrieval

Results from FAISS and BM25 are combined using Reciprocal Rank Fusion (RRF):

Score(d) = 1/(k + Rank_FAISS) + 1/(k + Rank_BM25)

where:

k = 60

This balances semantic relevance and keyword relevance.

---

## 🧠 Sanskrit-Specific Optimizations

### Unicode Normalization

Normalizes Sanskrit Unicode representations for consistent retrieval.

### Sandhi-Aware Chunking

Instead of splitting purely by character count, chunk boundaries consider:

* Danda (|)
* Double Danda (||)
* Paragraph Breaks
* Semantic Boundaries

This preserves the meaning of shlokas and verses.

### Transliteration Layer

Documents can be processed in:

* Devanagari
* IAST
* SLP1 (optional)

This improves retrieval accuracy and embedding quality.

---

## 📂 Project Structure

```text
RAG_Sanskrit_SanketMaraskolhe/

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

└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository_url>
cd RAG_Sanskrit_SanketMaraskolhe
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r code/requirements.txt
```

---

## 📥 Add Documents

Place Sanskrit files inside:

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

## 🔨 Build Index

```bash
python -m code.src.pipeline.rag_pipeline
```

Artifacts generated:

```text
artifacts/faiss_index/
artifacts/bm25_index/
artifacts/logs/
artifacts/metrics/
```

---

## 🌐 Launch Application

```bash
streamlit run code/app.py
```

Default URL:

```text
http://localhost:8501
```

---

## 📊 Performance Characteristics

| Component             | Purpose            |
| --------------------- | ------------------ |
| FAISS                 | Semantic Retrieval |
| BM25                  | Keyword Retrieval  |
| TinyLlama             | Answer Generation  |
| Streamlit             | User Interface     |
| Sentence Transformers | Embedding Creation |

### CPU Optimization

* 4-bit Quantization
* GGUF Model Format
* Low RAM Consumption
* Multi-threaded Inference
* CPU-Only Deployment

---

## 🛠 Tech Stack

* Python
* Streamlit
* FAISS
* BM25
* TinyLlama
* Sentence Transformers
* NumPy
* Pandas
* llama.cpp

---

## 🔮 Future Enhancements

* OCR Support for Sanskrit Manuscripts
* Hybrid Re-ranking Models
* Multi-language Retrieval
* Agentic RAG Workflows
* Knowledge Graph Integration
* API Deployment with FastAPI
* Docker Support

---

## 📜 License

This project is developed for research, educational, and Sanskrit language understanding purposes.

---

## 👨‍💻 Author

Prashant Ashok Darshanwar

AI/ML Engineer | GenAI Developer | RAG Systems Builder
