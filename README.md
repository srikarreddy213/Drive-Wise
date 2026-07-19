# 🚗 DriveWise — Metadata-Aware Automotive RAG Assistant

An AI chatbot that answers questions about car brochures using **Retrieval-Augmented Generation (RAG)** with **metadata filtering**, orchestrated by **LangGraph** and powered by **Groq**.

## Features
- 📄 Upload any car brochure PDF and index it in seconds
- 🎯 Metadata-filtered search — queries only the selected car's chunks
- 🔍 Composite re-ranking (semantic + keyword + section signal)
- 💬 Grounded answers with page number & section citations
- ⚡ Lightning-fast response times (Groq-powered)
- 📊 Automated Evaluation (Ragas + LLM-as-a-Judge fallback)
- 🕒 Asynchronous evaluation background execution for lag-free UI
- 📈 Analytics dashboard with query logs and eval score trends

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key in the environment
echo GROQ_API_KEY=your_key_here > .env

# 3. Run
streamlit run app.py
```

## Usage
1. Open **http://localhost:8501**
2. Make sure your `GROQ_API_KEY` is set in the `.env` file.
3. Select your car from the sidebar or upload a brochure in the **📄 Ingest Brochure** tab.
4. Ask any question in **💬 Chat Assistant**! If details are missing from the brochure, the assistant will seamlessly use general expert knowledge to answer your question.

## Architecture
```
User Query
    │
    ▼
Metadata Filter (brand + model pre-filter)
    │
    ▼
FAISS Vector Search (top-10 cosine-similar chunks)
    │
    ▼
Composite Re-Ranking (semantic + keyword + section)
    │
    ▼
Groq LLM (grounded answer with citations + fallback knowledge)
    │
    ▼
Ragas Evaluation (async background thread)
    │
    ▼
SQLite Logging → Analytics Dashboard
```

## Tech Stack
| Component | Technology |
|---|---|
| Orchestration | **LangChain** + **LangGraph** |
| UI | Streamlit |
| PDF Parsing | PyPDF |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector DB | FAISS CPU (IndexFlatIP) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Evaluation | **Ragas** (Fallback: LLM-as-a-Judge Groq) |
| Logging | SQLite |
| Config | python-dotenv |

## Project Structure
```
drivewise/
├── app.py               ← Streamlit UI (3 tabs)
├── requirements.txt
├── .env                 ← GROQ_API_KEY (never commit!)
├── src/
│   ├── config.py        ← All settings & prompt templates
│   ├── ingestion.py     ← PDF → metadata-tagged chunks
│   ├── vector_store.py  ← FAISS index management
│   ├── evaluator.py     ← LLM-as-a-judge scoring
│   ├── rag_graph.py     ← LangGraph compilation and pipeline steps
│   └── logger.py        ← SQLite query logging
└── data/                ← Saved database and vectorstore
    ├── faiss_index/     ← Pre-built vector index files (index.faiss, index.pkl)
    └── logs.db          ← Created dynamically at runtime
```

