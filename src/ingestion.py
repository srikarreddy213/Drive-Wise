"""
ingestion.py — LangChain-powered PDF ingestion pipeline for DriveWise.

Uses:
  • langchain_community.document_loaders.PyPDFLoader   — page-by-page loading
  • langchain.text_splitter.RecursiveCharacterTextSplitter — semantic chunking
  • Rule-based section classifier (keyword matching)

Each output Document carries metadata:
    {"car_brand", "car_model", "section", "page_number", "document_version"}
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    SECTION_KEYWORDS, BROCHURES_DIR,
)

# ── Section classifier (keyword-based) ────────────────────────────────────────

def _classify_section(text: str) -> str:
    tl = text.lower()
    scores = {
        sec: sum(1 for kw in kws if kw in tl)
        for sec, kws in SECTION_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Information"


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_brochure(
    pdf_path: str,
    car_brand: str,
    car_model: str,
    document_version: str,
) -> list[Document]:
    """
    Load a PDF and return a list of LangChain Document objects,
    each enriched with DriveWise metadata.

    Pipeline:
        PyPDFLoader → RecursiveCharacterTextSplitter → section classifier
        → metadata tagging → List[Document]
    """
    # 1. Load pages with LangChain
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()          # List[Document], one per page

    if not pages:
        raise ValueError(f"No pages extracted from: {pdf_path}")

    # 2. Split into semantic chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks: list[Document] = splitter.split_documents(pages)

    # 3. Classify section & inject metadata
    brand_t   = car_brand.strip().title()
    model_t   = car_model.strip().title()
    version_t = document_version.strip()

    tagged: list[Document] = []
    for doc in raw_chunks:
        if len(doc.page_content.strip()) < MIN_CHUNK_SIZE:
            continue

        section = _classify_section(doc.page_content)

        # page_number may come from PyPDFLoader metadata as 'page' (0-indexed)
        page_num = doc.metadata.get("page", 0) + 1

        doc.metadata.update({
            "car_brand":        brand_t,
            "car_model":        model_t,
            "section":          section,
            "page_number":      page_num,
            "document_version": version_t,
        })
        tagged.append(doc)

    print(
        f"[Ingestion] {len(pages)} pages -> {len(tagged)} chunks "
        f"({brand_t} {model_t} {version_t})"
    )
    return tagged


def save_uploaded_pdf(uploaded_file, car_brand: str, car_model: str) -> str:
    """Save a Streamlit UploadedFile to data/brochures/<brand>/. Returns the path."""
    brand_folder = BROCHURES_DIR / car_brand.lower().strip().replace(' ', '_').replace('-', '_')
    brand_folder.mkdir(parents=True, exist_ok=True)
    filename  = f"{car_brand.lower().strip()}_{car_model.lower().strip()}.pdf"
    save_path = brand_folder / filename
    with open(save_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return str(save_path)
