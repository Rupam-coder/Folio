<div align="center">

# 📖 Folio. — Production Document QA RAG Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq Cloud](https://img.shields.io/badge/Groq-Llama_3.1_8B-F05A28.svg?logo=openai&logoColor=white)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-In--Memory-FF6F00.svg)](https://www.trychroma.com/)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel_Ready-000000.svg?logo=vercel&logoColor=white)](https://vercel.com/)
[![Render](https://img.shields.io/badge/Backend-Render_Ready-46E3B7.svg?logo=render&logoColor=white)](https://render.com/)

An end-to-end, decoupled **Document Question-Answering (RAG)** application featuring a **FastAPI** backend with **Groq Llama 3.1 8B** ultra-fast inference and a minimalist, tactile **"Folio." Editorial Research Studio** frontend inspired by Anthropic Claude, Kinfolk, and Notion.

[Monorepo Architecture](#-monorepo-architecture) • [Live Benchmark Findings](#-live-benchmark-findings-dataeval_reportjson) • [Backend API](#-backend-api-documentation) • [Frontend Client](#-frontend-design--vercel-deployment) • [Local Development](#-local-development-quickstart) • [Production Deployment](#-production-deployment-guide)

---

</div>

## 🏛️ Monorepo Architecture

The repository is organized into three distinct, decoupled directories:

```
rag-eval-port/
├── backend/                  # FastAPI REST Service (Deployable to Render / Railway)
│   ├── main.py               # Complete API with /upload, /chat, and Groq Llama 3.1
│   ├── requirements.txt      # Production Python dependencies
│   ├── .env.example          # Sample environment variables (GROQ_API_KEY)
│   └── .gitignore            # Backend-specific ignore rules
│
├── frontend/                 # "Folio." Minimalist Client (Deployable to Vercel)
│   ├── index.html            # Tactile editorial workbench with Marked.js & Tailwind
│   ├── vercel.json           # Vercel SPA routing and cleanUrls configuration
│   └── .gitignore            # Frontend-specific ignore rules
│
├── data/                     # Evaluation Corpus, Reports & Test Assets
│   ├── eval_dataset.json     # 20-sample ground-truth query and passage corpus
│   ├── eval_report.json      # Quantitative benchmark findings & rank telemetry
│   └── sample_document.pdf   # Pre-generated PDF test document
│
├── .gitignore                # Root repository ignore rules (.venv, node_modules)
└── README.md                 # Complete system documentation
```

---

## 📊 Live Benchmark Findings (`data/eval_report.json`)

To validate retrieval accuracy before generating responses, the retriever was benchmarked against the 20-sample question and ground-truth document corpus in [`data/eval_dataset.json`](file:///e:/rag-eval-port/data/eval_dataset.json).

### 🏆 Overall Retrieval Telemetry

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Total Queries Evaluated** | **20** | Representative ML, NLP, and RAG retrieval inquiries |
| **Retrieval Hit Rate (Hit@4)** | **75.00%** | Ground-truth passage retrieved within top-4 candidate chunks |
| **Top-1 Hit Rate (Hit@1)** | **45.00%** | Ground-truth passage ranked as the #1 most relevant chunk |
| **Mean Reciprocal Rank (MRR)** | **0.5792** | Average reciprocal rank ($\frac{1}{N} \sum \frac{1}{\text{rank}}$) across all queries |
| **Average Query Latency** | **~310ms** | Vector proximity search + Groq Llama 3.1 streaming inference |

```
📈 Retrieval Performance Visualization:
Top-4 Hit Rate (Hit@4)   [███████████████     ]  75.00% (15 / 20 queries)
Top-1 Hit Rate (Hit@1)   [█████████           ]  45.00% (9 / 20 queries)
Mean Reciprocal Rank     [███████████         ]  0.5792 MRR
```

### 📋 Per-Query Rank Breakdown (`data/eval_report.json`)

| ID | Query Excerpt | Hit? | Rank | Retrieved Ground-Truth Source | Page |
| :---: | :--- | :---: | :---: | :--- | :---: |
| `q01` | What is the primary function of attention in transformers? | ✅ YES | **1** | `transformer_architecture.pdf` | Page 3 |
| `q02` | How does dense vector retrieval differ from sparse BM25? | ✅ YES | **2** | `hybrid_search.pdf` | Page 9 |
| `q03` | Why is document chunk overlap recommended for RAG? | ✅ YES | **1** | `rag_best_practices.pdf` | Page 2 |
| `q04` | What role does ChromaDB play in a local RAG architecture? | ✅ YES | **2** | `rag_troubleshooting.pdf` | Page 8 |
| `q05` | What is the purpose of temperature parameter in LLMs? | ✅ YES | **3** | `context_engineering.pdf` | Page 4 |
| `q06` | How does cross-encoder reranking improve retrieval quality? | ❌ NO | — | `context_engineering.pdf` | Page 4 |
| `q07` | What are common failure modes of naive RAG systems? | ✅ YES | **1** | `rag_troubleshooting.pdf` | Page 8 |
| `q08` | What metric does HNSW optimize during vector indexing? | ✅ YES | **4** | `advanced_retrievers.pdf` | Page 5 |
| `q09` | How does recursive character splitting decide where to split?| ❌ NO | — | `advanced_retrievers.pdf` | Page 5 |
| `q10` | What is cosine similarity and how is it calculated? | ✅ YES | **1** | `linear_algebra_for_ml.pdf` | Page 3 |
| `q11` | What is the advantage of using sentence-transformers? | ❌ NO | — | `linear_algebra_for_ml.pdf` | Page 3 |
| `q12` | How does hybrid search combine lexical and dense retrieval? | ✅ YES | **2** | `ir_fundamentals.pdf` | Page 5 |
| `q13` | What is hallucination in generative language models? | ✅ YES | **1** | `llm_evaluation.pdf` | Page 5 |
| `q14` | Why is metadata filtering useful during vector retrieval? | ✅ YES | **1** | `vector_filtering.pdf` | Page 2 |
| `q15` | What is context stuffing and why can it degrade performance? | ❌ NO | — | `rag_troubleshooting.pdf` | Page 8 |
| `q16` | How does query rewriting assist in conversational RAG? | ❌ NO | — | `context_engineering.pdf` | Page 4 |
| `q17` | What is normalized discounted cumulative gain (NDCG)? | ✅ YES | **1** | `ir_evaluation_metrics.pdf` | Page 6 |
| `q18` | What role does tokenization play before text chunking? | ✅ YES | **2** | `llm_evaluation.pdf` | Page 5 |
| `q19` | What is parent document retrieval in LangChain? | ✅ YES | **1** | `advanced_retrievers.pdf` | Page 5 |
| `q20` | How does quantization reduce embedding memory footprint? | ✅ YES | **1** | `vector_optimization.pdf` | Page 7 |

---

## ⚙️ Backend API Documentation

The backend service is located in [`backend/main.py`](file:///e:/rag-eval-port/backend/main.py) and exposes high-throughput async endpoints.

### 1. Document Ingestion (`POST /upload`)
Accepts a PDF document via multipart form upload, extracts text with `PyPDFLoader`, splits documents into 500-character chunks with 50-character overlap, and stores them in an in-memory Chroma vector collection.

* **Request**: `multipart/form-data` with `file: UploadFile`
* **Response**:
```json
{
  "session_id": "3d76c54c731a4713981ee49223ee2d95",
  "filename": "sample_document.pdf",
  "chunk_count": 2,
  "page_count": 2,
  "message": "Document successfully processed and indexed into in-memory Chroma vectorstore."
}
```

### 2. Grounded Question Answering (`POST /chat`)
Retrieves the top-4 most similar vector chunks for the given `session_id`, normalizes line breaks and whitespace, formats context blocks with source document names and 1-based page numbers, and streams the prompt to **Groq's Llama 3.1 8B** model.

* **Request**:
```json
{
  "session_id": "3d76c54c731a4713981ee49223ee2d95",
  "query": "What is the Groq LPU throughput?",
  "top_k": 4
}
```
* **Response**:
```json
{
  "session_id": "3d76c54c731a4713981ee49223ee2d95",
  "answer": "The Groq LPU delivers deterministic execution without dynamic scheduling, achieving throughput exceeding 500 tokens per second. [Source: sample_document.pdf, Page 1]",
  "sources": [
    {
      "chunk_id": 1,
      "source": "sample_document.pdf",
      "page": 1,
      "content": "Groq LPU Architecture and Specifications. The Groq Language Processing Unit is an AI hardware architecture..."
    }
  ]
}
```

### 3. Service Telemetry (`GET /`)
Returns active session count and service metadata.

---

## 🎨 Frontend Design & Vercel Deployment

The frontend in [`frontend/index.html`](file:///e:/rag-eval-port/frontend/index.html) is built as a self-contained, zero-dependency SPA with the warm **"Folio." Editorial Research Studio** aesthetic:

* **Warm Editorial Palette**: Archival Linen (`#F4F0EA`), Crisp Ivory Paper (`#FFFDF9`), Deep Espresso Ink (`#1C1917`), and Burnt Terracotta (`#D9532F`) accents.
* **Refined Typography**: Editorial headers in `'Instrument Serif'`, clean reading body in `'Plus Jakarta Sans'`, and page badges in `'Space Mono'`.
* **Ambient Lighting Spotlight**: A subtle cursor-responsive ambient spotlight following the cursor across panels.
* **Terracotta Laser Scanner**: A vertical laser sweep animation active while a document is being processed.
* **Interactive Citation Linking**: Clicking any `⌗ PAGE X` citation pill in the chat smoothly scrolls and pulses the matching source card in the right drawer.
* **Normalized Source Cards**: Removes broken line breaks and word stacking; text flows as clean, readable paragraphs with click-to-expand toggles.

### Dynamic Backend Routing
The frontend dynamically chooses its backend target:
```javascript
const BACKEND_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
  ? "http://localhost:8000" 
  : "YOUR_RENDER_BACKEND_URL";
```

---

## 🚀 Local Development Quickstart

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### 2. Set Up Backend
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The FastAPI backend will be live at `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`).

### 3. Launch Frontend Client
In a separate terminal:
```bash
# Navigate to frontend directory
cd frontend

# Serve with lightweight Python HTTP server
python -m http.server 3000
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser (or double-click `frontend/index.html`).

---

## 🌐 Production Deployment Guide

### Deploying Backend to Render
1. Push this repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/) and create a **New Web Service**.
3. Select your repository and configure:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variable:
   - `GROQ_API_KEY`: `your_groq_api_key_here`
5. Click **Create Web Service** and copy your generated Render URL (e.g., `https://folio-rag-api.onrender.com`).

### Deploying Frontend to Vercel
1. In [`frontend/index.html`](file:///e:/rag-eval-port/frontend/index.html), update `YOUR_RENDER_BACKEND_URL`:
   ```javascript
   const BACKEND_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
     ? "http://localhost:8000" 
     : "https://folio-rag-api.onrender.com";
   ```
2. Log into [Vercel](https://vercel.com/) and import the repository.
3. Configure the project:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Other`
4. Click **Deploy**. Vercel will use [`frontend/vercel.json`](file:///e:/rag-eval-port/frontend/vercel.json) to serve the SPA with clean URLs.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
