"""
vector_store.py — LangChain FAISS vector store for DriveWise.

Uses:
  • langchain_huggingface.HuggingFaceEmbeddings  — local all-MiniLM-L6-v2
  • langchain_community.vectorstores.FAISS       — IndexFlatL2 under the hood

Key design: metadata is stored inside each LangChain Document, so metadata
filtering is done in Python before a similarity search over the subset.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    TOP_K_RAW,
)

_FAISS_PATH    = str(FAISS_INDEX_DIR)
_DOCS_PKL      = FAISS_INDEX_DIR / "docs.pkl"

# ── Singleton embeddings model ─────────────────────────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Build ─────────────────────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document]) -> FAISS:
    """Create a brand-new LangChain FAISS store from a Document list."""
    print(f"[VectorStore] Building FAISS store from {len(docs)} documents ...")
    store = FAISS.from_documents(docs, get_embeddings())
    print(f"[VectorStore] Store built: {store.index.ntotal} vectors")
    return store


def add_documents_to_store(
    new_docs: list[Document],
    store: FAISS,
) -> FAISS:
    """Append new documents to an existing FAISS store."""
    store.add_documents(new_docs)
    print(f"[VectorStore] Added {len(new_docs)} docs. Total: {store.index.ntotal}")
    return store


# ── Persist ───────────────────────────────────────────────────────────────────

def save_vectorstore(store: FAISS) -> None:
    """Save FAISS index + docstore to data/faiss_index/."""
    store.save_local(_FAISS_PATH)
    print(f"[VectorStore] Saved {store.index.ntotal} vectors to {_FAISS_PATH}")


def load_vectorstore() -> FAISS | None:
    """Load FAISS store from disk. Returns None if not found."""
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        print("[VectorStore] No saved store — starting fresh.")
        return None
    store = FAISS.load_local(
        _FAISS_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    print(f"[VectorStore] Loaded store: {store.index.ntotal} vectors")
    return store


# ── Metadata-filtered search ──────────────────────────────────────────────────

def search_store(
    query: str,
    car_brand: str,
    car_model: str,
    store: FAISS,
    top_k: int = TOP_K_RAW,
) -> list[Document]:
    """
    Metadata-filtered similarity search.

    LangChain FAISS supports `filter` kwarg for metadata:
        store.similarity_search(query, k=k, filter={...})

    We normalise brand/model to Title-case (matching ingestion tagging).
    """
    brand_n = car_brand.strip().title()
    model_n = car_model.strip().title()

    results = store.similarity_search_with_score(
        query,
        k=top_k,
        filter={"car_brand": brand_n, "car_model": model_n},
        fetch_k=2000,
    )

    # results = List[(Document, float_score)]
    # Attach score to metadata so the retriever can use it
    docs = []
    for doc, score in results:
        doc.metadata["similarity_score"] = float(score)
        docs.append(doc)

    if not docs:
        print(f"[VectorStore] No documents found for '{brand_n} {model_n}'")
    return docs


# ── Utility ───────────────────────────────────────────────────────────────────

def get_available_cars(store: FAISS | None) -> dict[str, list[str]]:
    """
    Inspect the docstore to return {brand: [model, ...]} for UI dropdowns.
    Works by iterating over all stored documents' metadata.
    """
    if store is None:
        return {}
    cars: dict[str, list[str]] = {}
    for doc_id, doc in store.docstore._dict.items():
        b = doc.metadata.get("car_brand", "Unknown")
        m = doc.metadata.get("car_model", "Unknown")
        cars.setdefault(b, [])
        if m not in cars[b]:
            cars[b].append(m)
    return cars
