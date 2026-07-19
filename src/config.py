"""
config.py — Centralized configuration for DriveWise (LangChain + LangGraph).

All paths, model identifiers, chunking parameters, retrieval settings,
section taxonomy, and LangChain PromptTemplate strings live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
BROCHURES_DIR   = DATA_DIR / "brochures"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
LOGS_DB_PATH    = DATA_DIR / "logs.db"

BROCHURES_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# ─── Model identifiers ────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # HuggingFaceEmbeddings
GROQ_MODEL_NAME      = "llama-3.3-70b-versatile"  # ChatGroq

# ─── Chunking (LangChain RecursiveCharacterTextSplitter) ─────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RAW   = 10    # docs fetched from FAISS before re-ranking
TOP_K_FINAL = 3     # docs passed to LLM

# ─── Re-Ranking weights ───────────────────────────────────────────────────────
WEIGHT_SEMANTIC = 0.5
WEIGHT_KEYWORD  = 0.3
WEIGHT_SECTION  = 0.2

# ─── Section taxonomy ─────────────────────────────────────────────────────────
SECTIONS = [
    "Engine & Performance",
    "Safety Features",
    "Comfort & Convenience",
    "Infotainment & Connectivity",
    "Exterior Design",
    "Interior Design",
    "Dimensions & Capacity",
    "Fuel Efficiency",
    "Variants & Pricing",
    "Warranty & Service",
    "General Information",
]

SECTION_KEYWORDS: dict[str, list[str]] = {
    "Engine & Performance": [
        "engine","torque","horsepower","hp","bhp","cc","displacement","rpm",
        "transmission","gearbox","turbo","cylinder","petrol","diesel","hybrid",
        "electric","motor","power","speed","acceleration","0-100","drivetrain",
        "awd","4wd","rwd","fwd","clutch","dct","imt","amt",
    ],
    "Safety Features": [
        "airbag","abs","ebd","esc","esp","traction control","isofix","seatbelt",
        "hill hold","hill descent","blind spot","collision","lane departure",
        "pedestrian detection","emergency brake","ncap","safety","crash",
        "rollover","curtain","rear camera","parking sensor","360 camera",
    ],
    "Comfort & Convenience": [
        "air conditioning","climate control","ac","ventilated seat","heated seat",
        "sunroof","moonroof","panoramic","wireless charging","usb","armrest",
        "lumbar","push button start","keyless","remote start","ambient light",
        "cruise control","auto park","cooled seat",
    ],
    "Infotainment & Connectivity": [
        "infotainment","touchscreen","android auto","apple carplay","bluetooth",
        "wifi","4g","lte","navigation","gps","speakers","sound system","jbl",
        "bose","harman","display","instrument cluster","digital cockpit",
        "connected car","ota","voice assistant","inch screen",
    ],
    "Exterior Design": [
        "grille","headlamp","taillight","bumper","alloy wheel","tyre","tire",
        "wheel","roof rail","spoiler","mirror","door handle","chrome","colour",
        "color","paint","exterior","led","drl","fog lamp","puddle lamp",
    ],
    "Interior Design": [
        "upholstery","leather","fabric","cabin","dashboard","steering wheel",
        "instrument panel","interior","trim","material","wood","carbon fibre",
        "piano black","soft touch","door pad","flat bottom","d-cut",
    ],
    "Dimensions & Capacity": [
        "length","width","height","wheelbase","ground clearance","boot","trunk",
        "luggage","litre","liter","seating capacity","passengers","turning radius",
        "weight","kerb weight","gross vehicle","mm",
    ],
    "Fuel Efficiency": [
        "mileage","fuel efficiency","kmpl","mpg","fuel economy","range","battery",
        "kwh","charging","fuel tank","tank capacity","arai","wltp","consumption",
    ],
    "Variants & Pricing": [
        "variant","trim","base","mid","top","price","ex-showroom","on-road",
        "emi","cost","version","grade","specification","optional","standard",
    ],
    "Warranty & Service": [
        "warranty","guarantee","service","maintenance","roadside assistance",
        "kilometre","kilometer","service interval","free service","extended warranty",
    ],
}

# ─── LangChain prompt templates (as raw strings) ──────────────────────────────
SYSTEM_PROMPT = """\
You are DriveWise, an expert automotive assistant. Answer questions about car brochures and vehicles accurately.

RULES:
1. Primary Source: Use the provided brochure context to answer the question. Include page numbers and section citations if available.
2. Fallback Source: If the details are not found in the brochure context (e.g. if the context is empty or missing details like specific mileage, prices, or comparisons), you MUST use your own expert automotive knowledge to answer the question about the specific car model.
3. Transparency: When using fallback general knowledge, clearly prefix or suffix that part of the answer with a note stating that this detail is from general vehicle specifications, as it was not detailed in the uploaded brochure (e.g. "*(Note: This information is from general vehicle specifications, as it was not detailed in the brochure)*").
4. Be specific: include numbers, units, and exact feature names.
5. Use bullet points for feature lists.
6. Never refuse to answer. If the details are not in the context, provide the best known specifications for the selected car.
"""

RAG_PROMPT_TEMPLATE = """\
BROCHURE CONTEXT for {car_brand} {car_model}:

{context}

---
USER QUESTION: {question}

Answer using the brochure context if available. If not, use your general knowledge of the {car_brand} {car_model}.
"""

EVAL_CR_TEMPLATE = """\
Rate whether the retrieved context is relevant to the user query (0 to 1).
USER QUERY: {query}
RETRIEVED CONTEXT: {context}
- 1.0 = directly answers the query
- 0.5 = partially relevant
- 0.0 = not relevant
Respond ONLY with JSON: {{"score": 0.8, "reason": "brief explanation"}}
"""

EVAL_FAITH_TEMPLATE = """\
Rate whether the generated answer is faithful to the context (0 to 1).
CONTEXT: {context}
GENERATED ANSWER: {answer}
- 1.0 = every claim supported by context
- 0.5 = mostly supported
- 0.0 = hallucination detected
Respond ONLY with JSON: {{"score": 0.9, "reason": "brief explanation"}}
"""

EVAL_AC_TEMPLATE = """\
Rate whether the answer correctly addresses the question (0 to 1).
USER QUERY: {query}
CONTEXT: {context}
GENERATED ANSWER: {answer}
- 1.0 = fully and correctly answers
- 0.5 = partially answers
- 0.0 = wrong or irrelevant
Respond ONLY with JSON: {{"score": 0.7, "reason": "brief explanation"}}
"""

# Aliases and templates for generator.py and evaluator.py
USER_PROMPT_TEMPLATE = """\
BROCHURE CONTEXT for {car_brand} {car_model}:

{context_blocks}

---
USER QUESTION: {user_query}

Answer using ONLY the brochure context above.
"""

CONTEXT_RELEVANCE_PROMPT = EVAL_CR_TEMPLATE
FAITHFULNESS_PROMPT = EVAL_FAITH_TEMPLATE
ANSWER_CORRECTNESS_PROMPT = EVAL_AC_TEMPLATE

