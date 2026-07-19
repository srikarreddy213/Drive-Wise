"""
rag_graph.py — LangGraph-powered RAG pipeline for DriveWise.

Architecture (StateGraph)
─────────────────────────

  [START]
     │
     ▼
  retrieve          ← FAISS metadata-filtered search → top-10 docs
     │
     ▼
  rerank            ← composite score (semantic + keyword + section)
     │
     ├─── no_docs ──► no_answer   ← respond "not found"
     │
     ▼
  generate          ← ChatGoogleGenerativeAI with grounded prompt
     │
     ▼
  evaluate          ← Ragas (Faithfulness + Answer Relevancy) with
     │                 LLM-as-a-Judge fallback
     ▼
  [END]

Every node receives and returns a typed GraphState (TypedDict).
The final state contains the answer, sources, eval scores, and status.
"""

import re
import json
import warnings
import sys
import types

# ── Ragas / LangChain compatibility patch ─────────────────────────────────────
# Ragas 0.2.x internally does:
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# which fails on newer langchain_community versions where the module was
# removed.  We inject a stub so the import succeeds.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        v_module = types.ModuleType("langchain_community.chat_models.vertexai")
        try:
            from langchain_google_vertexai import ChatVertexAI as _RealChatVertexAI
            v_module.ChatVertexAI = _RealChatVertexAI
        except ImportError:
            # langchain-google-vertexai not installed — create a harmless stub
            class _StubChatVertexAI:
                """Stub so that ``from … import ChatVertexAI`` doesn't crash."""
            v_module.ChatVertexAI = _StubChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = v_module
    except Exception:
        pass

warnings.filterwarnings("ignore")

from typing import TypedDict, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

from src.config import (
    GROQ_MODEL_NAME,
    GROQ_API_KEY,
    TOP_K_RAW,
    TOP_K_FINAL,
    WEIGHT_SEMANTIC,
    WEIGHT_KEYWORD,
    WEIGHT_SECTION,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    SECTION_KEYWORDS,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. State definition
# ══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    # inputs
    query:      str
    car_brand:  str
    car_model:  str
    api_key:    str
    store:      FAISS          # passed through, not modified by graph
    skip_eval:  bool           # whether to skip evaluation

    # intermediate
    raw_docs:       list[Document]   # after retrieval
    reranked_docs:  list[Document]   # after re-ranking
    context:        str              # formatted context string

    # outputs
    answer:            str
    sources:           list[dict]
    status:            str           # "SUCCESS" | "NO_ANSWER_FOUND"
    eval_scores:       dict          # CR, Faithfulness, AC


# ══════════════════════════════════════════════════════════════════════════════
# 2. Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

_STOP = {
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "have","has","had","do","does","did","will","would","could","should",
    "of","in","to","for","on","with","at","by","from","that","this","it",
    "what","who","when","where","how","all","some","not","no","if","just",
    "i","me","my","we","our","you","your","they","them",
}

def _kw(text: str) -> set[str]:
    """Extract keywords (≥3 chars) from *text*, excluding stop words."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP}

def _kw_overlap(query: str, text: str) -> float:
    """Fraction of query keywords that also appear in *text*."""
    qk = _kw(query)
    return len(qk & _kw(text)) / len(qk) if qk else 0.0

def _section_bonus(query: str, section: str) -> float:
    """Return 1.0 if the query seems related to *section*, else 0.0."""
    q = query.lower()
    for word in section.lower().split():
        if len(word) > 3 and word in q:
            return 1.0
    synonyms = {
        "safety features":             ["airbag","abs","safe","crash","protect","ncap"],
        "engine & performance":        ["engine","power","speed","torque","hp","bhp","cc"],
        "fuel efficiency":             ["mileage","fuel","kmpl","economy","range","tank"],
        "infotainment & connectivity": ["screen","bluetooth","android","apple","audio"],
        "comfort & convenience":       ["comfort","seat","sunroof","climate","keyless"],
        "variants & pricing":          ["price","cost","variant","trim","version"],
        "dimensions & capacity":       ["size","dimension","boot","space","capacity"],
    }
    for key, syns in synonyms.items():
        if key in section.lower() and any(s in q for s in syns):
            return 1.0
    return 0.0

def _format_context(docs: list[Document]) -> str:
    """Build a numbered context string from re-ranked documents."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        blocks.append(
            f"[SOURCE {i}]\n"
            f"Section : {m.get('section','Unknown')}\n"
            f"Page    : {m.get('page_number','?')}\n"
            f"---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)

def _extract_sources(docs: list[Document]) -> list[dict]:
    """Return a compact citation list from the re-ranked documents."""
    sources = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        preview = doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else "")
        sources.append({
            "source_number":    i,
            "page_number":      m.get("page_number", "?"),
            "section":          m.get("section", "Unknown"),
            "car_brand":        m.get("car_brand", ""),
            "car_model":        m.get("car_model", ""),
            "document_version": m.get("document_version", ""),
            "text_preview":     preview,
        })
    return sources

def _parse_eval_json(raw: str) -> dict:
    """Extract ``{\"score\": …, \"reason\": …}`` from an LLM text response."""
    try:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score":  float(data.get("score", 0.0)),
                "reason": str(data.get("reason", "—")),
            }
    except Exception:
        pass
    return {"score": 0.0, "reason": "Parse failed"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Graph nodes
# ══════════════════════════════════════════════════════════════════════════════

def node_retrieve(state: RAGState) -> dict:
    """
    Node: RETRIEVE
    Uses LangChain FAISS with metadata filter to fetch TOP_K_RAW candidates.
    """
    from src.vector_store import search_store
    docs = search_store(
        state["query"],
        state["car_brand"],
        state["car_model"],
        state["store"],
        top_k=TOP_K_RAW,
    )
    print(f"[Graph:Retrieve] {len(docs)} raw docs")
    return {"raw_docs": docs}


def node_rerank(state: RAGState) -> dict:
    """
    Node: RERANK
    Composite re-ranking: semantic similarity + keyword overlap + section bonus.
    Returns top TOP_K_FINAL documents in `reranked_docs`.
    """
    raw = state["raw_docs"]
    query = state["query"]

    if not raw:
        return {"reranked_docs": [], "context": ""}

    sims = [d.metadata.get("similarity_score", 0.0) for d in raw]
    lo, hi = min(sims), max(sims)
    span = hi - lo if hi != lo else 1.0

    scored = []
    for doc in raw:
        norm_sim = (doc.metadata.get("similarity_score", 0.0) - lo) / span
        kw       = _kw_overlap(query, doc.page_content)
        sec      = _section_bonus(query, doc.metadata.get("section", ""))
        score    = WEIGHT_SEMANTIC * norm_sim + WEIGHT_KEYWORD * kw + WEIGHT_SECTION * sec
        doc.metadata["composite_score"] = round(score, 4)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:TOP_K_FINAL]]
    context = _format_context(top)
    print(f"[Graph:Rerank] Top {len(top)} docs selected")
    return {"reranked_docs": top, "context": context}


def node_no_answer(state: RAGState) -> dict:
    """
    Node: NO_ANSWER
    Reached when no relevant docs were found for the selected car.
    """
    return {
        "answer":      ("I cannot find relevant information for this question. "
                        "Please make sure the correct brochure has been uploaded "
                        f"for {state['car_brand']} {state['car_model']}."),
        "sources":     [],
        "status":      "NO_ANSWER_FOUND",
        "eval_scores": {},
    }


def node_generate(state: RAGState) -> dict:
    """
    Node: GENERATE
    Uses ChatGoogleGenerativeAI with a grounded RAG prompt.
    LangChain handles the system/human message structure.
    """
    api_key = state.get("api_key") or GROQ_API_KEY
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=api_key,
        temperature=0.2,
    )

    # Build the prompt with LangChain ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  RAG_PROMPT_TEMPLATE),
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "car_brand": state["car_brand"],
            "car_model": state["car_model"],
            "context":   state["context"],
            "question":  state["query"],
        })
        answer = response.content.strip()
    except Exception as exc:
        answer = f"⚠️ Generation error: {exc}"

    no_answer_signals = [
        "cannot find this information",
        "i cannot find",
        "not mentioned in the",
        "not available in the provided",
    ]
    status = (
        "NO_ANSWER_FOUND"
        if any(s in answer.lower() for s in no_answer_signals)
        else "SUCCESS"
    )

    sources = _extract_sources(state["reranked_docs"])
    print(f"[Graph:Generate] Status={status}")
    return {"answer": answer, "sources": sources, "status": status}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _run_ragas_evaluation(state: RAGState, api_key: str) -> dict | None:
    """
    Attempt Ragas-based evaluation.  Returns the eval_scores dict on
    success, or ``None`` if Ragas is unavailable / fails so the caller
    can fall back to LLM-as-a-Judge.
    """
    try:
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
    except ImportError:
        print("[Graph:Evaluate] Ragas not installed — falling back to LLM-as-a-Judge")
        return None

    try:
        llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            groq_api_key=api_key,
            temperature=0.0,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from src.vector_store import get_embeddings

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        # Build Ragas evaluation dataset
        sample = SingleTurnSample(
            user_input=state["query"],
            response=state["answer"],
            retrieved_contexts=[doc.page_content for doc in state["reranked_docs"]],
        )
        eval_dataset = EvaluationDataset(samples=[sample])

        print("[Graph:Evaluate] Running Ragas evaluation (Faithfulness, Answer Relevancy) ...")
        result = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
            raise_exceptions=False,
        )

        # result.scores is List[Dict[str, Any]]
        scores = result.scores[0] if result.scores else {}
        faith_score = float(scores.get("faithfulness", 0.0))
        ar_score = float(scores.get("answer_relevancy", 0.0))

        eval_scores = {
            "context_relevance":  {"score": ar_score, "reason": "Evaluated via Ragas Answer Relevancy"},
            "faithfulness":       {"score": faith_score, "reason": "Evaluated via Ragas Faithfulness"},
            "answer_correctness": {"score": round((faith_score + ar_score) / 2, 4),
                                   "reason": "Average of Faithfulness & Relevancy"},
        }
        print(f"[Graph:Evaluate] Ragas Faithfulness={faith_score:.2f} AnswerRelevancy={ar_score:.2f}")
        return eval_scores

    except Exception as e:
        print(f"[Graph:Evaluate] Ragas evaluation failed: {e}")
        return None


def _run_llm_judge_evaluation(state: RAGState, api_key: str) -> dict:
    """
    Fallback evaluation using Llama-3.3 on Groq as an LLM-as-a-Judge.
    Evaluates Context Relevance, Faithfulness, and Answer Correctness.
    """
    from src.config import EVAL_CR_TEMPLATE, EVAL_FAITH_TEMPLATE, EVAL_AC_TEMPLATE

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=api_key,
        temperature=0.0,
    )

    def _judge(prompt_text: str) -> dict:
        try:
            response = llm.invoke(prompt_text)
            return _parse_eval_json(response.content)
        except Exception as exc:
            print(f"[Graph:Evaluate] LLM judge call failed: {exc}")
            return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

    context = state["context"]
    query = state["query"]
    answer = state["answer"]

    print("[Graph:Evaluate] Running LLM-as-a-Judge evaluation ...")
    cr    = _judge(EVAL_CR_TEMPLATE.format(query=query, context=context))
    faith = _judge(EVAL_FAITH_TEMPLATE.format(context=context, answer=answer))
    ac    = _judge(EVAL_AC_TEMPLATE.format(query=query, context=context, answer=answer))

    print(f"[Graph:Evaluate] LLM-Judge CR={cr['score']:.2f} Faith={faith['score']:.2f} AC={ac['score']:.2f}")
    return {
        "context_relevance":  cr,
        "faithfulness":       faith,
        "answer_correctness": ac,
    }


def node_evaluate(state: RAGState) -> dict:
    """
    Node: EVALUATE
    Attempts Ragas-based scoring first (Faithfulness + Answer Relevancy).
    If Ragas is unavailable or fails, falls back to LLM-as-a-Judge
    (Context Relevance, Faithfulness, Answer Correctness via Groq).
    Runs only when generation was successful and skip_eval is False.
    """
    if state.get("status") == "NO_ANSWER_FOUND" or state.get("skip_eval"):
        print("[Graph:Evaluate] Skipping evaluation (skip_eval is True or no answer found)")
        return {"eval_scores": {}}

    api_key = state.get("api_key") or GROQ_API_KEY
    if not api_key:
        print("[Graph:Evaluate] Skipping evaluation (no API key)")
        return {"eval_scores": {}}

    # Try Ragas first, fall back to LLM-as-a-Judge
    eval_scores = _run_ragas_evaluation(state, api_key)
    if eval_scores is None:
        eval_scores = _run_llm_judge_evaluation(state, api_key)

    return {"eval_scores": eval_scores}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Routing condition
# ══════════════════════════════════════════════════════════════════════════════

def route_after_rerank(state: RAGState) -> Literal["generate", "no_answer"]:
    """Send to 'generate' if we have docs, else 'no_answer'."""
    return "generate" if state["reranked_docs"] else "no_answer"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Build + compile the graph
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology:
        START → retrieve → rerank → [generate | no_answer] → evaluate → END
                                      (conditional)
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("retrieve",  node_retrieve)
    graph.add_node("rerank",    node_rerank)
    graph.add_node("generate",  node_generate)
    graph.add_node("no_answer", node_no_answer)
    graph.add_node("evaluate",  node_evaluate)

    # Edges
    graph.add_edge(START,       "retrieve")
    graph.add_edge("retrieve",  "rerank")
    graph.add_conditional_edges("rerank", route_after_rerank)
    graph.add_edge("generate",  "evaluate")
    graph.add_edge("evaluate",  END)
    graph.add_edge("no_answer", END)

    compiled = graph.compile()
    print("[RAGGraph] Graph compiled successfully.")
    return compiled


# Singleton compiled graph
_rag_graph = None

def get_rag_graph():
    """Return the singleton compiled LangGraph (builds once per process)."""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ══════════════════════════════════════════════════════════════════════════════
# 6. Public invocation function
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(
    query: str,
    car_brand: str,
    car_model: str,
    store: FAISS,
    api_key: str = "",
    skip_eval: bool = False,
) -> dict:
    """
    Run the full LangGraph RAG pipeline and return the final state.

    Args:
        query:     User question
        car_brand: e.g. "Hyundai"
        car_model: e.g. "I20"
        store:     Loaded LangChain FAISS vectorstore
        api_key:   Gemini API key (falls back to .env)
        skip_eval: Whether to skip evaluation (scores will be empty)

    Returns:
        Final RAGState dict with keys:
            answer, sources, status, eval_scores, context, reranked_docs
    """
    graph = get_rag_graph()

    initial_state: RAGState = {
        "query":        query,
        "car_brand":    car_brand,
        "car_model":    car_model,
        "api_key":      api_key or GROQ_API_KEY,
        "store":        store,
        "raw_docs":     [],
        "reranked_docs": [],
        "context":      "",
        "answer":       "",
        "sources":      [],
        "status":       "",
        "eval_scores":  {},
        "skip_eval":    skip_eval,
    }

    final_state = graph.invoke(initial_state)
    return final_state
