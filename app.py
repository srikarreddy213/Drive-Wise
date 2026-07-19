"""
app.py — DriveWise Streamlit Frontend  (LangChain + LangGraph edition)

Run:  streamlit run app.py

Tabs
────
  💬 Chat Assistant   — LangGraph RAG pipeline (retrieve → rerank → generate → evaluate)
  📄 Ingest Brochure  — LangChain PyPDFLoader + RecursiveCharacterTextSplitter
  📊 Analytics        — SQLite query logs, eval scores, response-time charts
"""

import time
import threading
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import GROQ_API_KEY
from src.logger import (
    initialize_database, log_query, update_eval_scores,
    get_all_logs, get_summary_stats,
)
from src.ingestion import ingest_brochure, save_uploaded_pdf
from src.vector_store import (
    build_vectorstore, add_documents_to_store,
    save_vectorstore, load_vectorstore, get_available_cars,
)
from src.rag_graph import run_rag_pipeline

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriveWise — LangGraph RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0b1e 0%, #1a1540 50%, #0d1b2a 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #94a3b8; font-weight: 600; font-size: .95rem; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: #fff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff !important; border: none; border-radius: 10px;
    padding: .55rem 1.6rem; font-weight: 600;
    box-shadow: 0 4px 15px rgba(99,102,241,.3); transition: all .2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,.5); }

.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 16px !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; }

.user-bubble {
    background: rgba(99,102,241,.18); border-left: 4px solid #6366f1;
    padding: 12px 16px; border-radius: 0 12px 12px 0; margin: 10px 0; color: #e2e8f0;
}
.bot-bubble {
    background: rgba(16,185,129,.12); border-left: 4px solid #10b981;
    padding: 14px 18px; border-radius: 0 12px 12px 0; margin: 10px 0;
    color: #e2e8f0; line-height: 1.7;
}
.graph-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: .75rem; font-weight: 600; margin-right: 6px;
    background: rgba(99,102,241,.25); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,.4);
}
.info-box {
    background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.3);
    border-radius: 10px; padding: 14px 18px; color: #93c5fd; font-size: .9rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Startup ──────────────────────────────────────────────────────────────────
initialize_database()

# Get key from environment or secrets
temp_key = GROQ_API_KEY
if not temp_key:
    try:
        temp_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

for key, default in [
    ("vectorstore",     None),
    ("chat_history",    []),
    ("available_cars",  {}),
    ("api_key",         temp_key),
    ("selected_brand",  ""),
    ("selected_model",  ""),
    ("last_ingested_info", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load saved vectorstore on first run
if st.session_state.vectorstore is None:
    vs = load_vectorstore()
    st.session_state.vectorstore   = vs
    st.session_state.available_cars = get_available_cars(vs)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#6366f1;margin-bottom:0'>🚗 DriveWise</h2>"
        "<p style='color:#64748b;font-size:.8rem;margin-top:2px'>"
        "Powered by LangChain + LangGraph</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 🚙 Select Car")
    cars = st.session_state.available_cars

    if not cars:
        st.markdown(
            '<div class="info-box">No brochures indexed yet.<br>'
            "Upload one in the <b>📄 Ingest</b> tab.</div>",
            unsafe_allow_html=True,
        )
        selected_brand = selected_model = ""
    else:
        brands_list = list(cars.keys())
        
        # Ensure st.session_state.selected_brand is valid
        if st.session_state.selected_brand not in brands_list:
            st.session_state.selected_brand = brands_list[0]
            
        selected_brand = st.selectbox(
            "Brand", 
            brands_list, 
            index=brands_list.index(st.session_state.selected_brand)
        )
        st.session_state.selected_brand = selected_brand

        models_list = cars.get(selected_brand, [])
        
        # Ensure st.session_state.selected_model is valid
        if st.session_state.selected_model not in models_list:
            st.session_state.selected_model = models_list[0] if models_list else ""
            
        selected_model = st.selectbox(
            "Model", 
            models_list, 
            index=models_list.index(st.session_state.selected_model) if st.session_state.selected_model in models_list else 0
        )
        st.session_state.selected_model = selected_model

    st.divider()

    # Graph topology info
    st.markdown("**LangGraph Pipeline**")
    for label in ["retrieve","rerank","generate","evaluate"]:
        st.markdown(f"<span class='graph-badge'>{label}</span>", unsafe_allow_html=True)

    st.divider()
    stats = get_summary_stats()
    ca, cb = st.columns(2)
    ca.metric("Queries",  stats["total_queries"])
    cb.metric("Avg Time", f"{stats['avg_response_time']}s")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_analytics = st.tabs(
    ["💬 Chat Assistant", "📄 Ingest Brochure", "📊 Analytics"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("<h2 style='color:#e2e8f0'>💬 Ask About Your Car</h2>", unsafe_allow_html=True)

    if not selected_brand or not selected_model:
        st.warning("⚠️ Select a car from the sidebar or upload a brochure first.")
    else:
        st.markdown(
            f"<p style='color:#64748b'>Querying: "
            f"<b style='color:#6366f1'>{selected_brand} {selected_model}</b> "
            f"&nbsp;|&nbsp; Pipeline: "
            f"<span class='graph-badge'>LangGraph</span></p>",
            unsafe_allow_html=True,
        )

        # Render conversation
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">👤 <b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble">🚗 <b>DriveWise:</b><br><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander("📑 View Sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f"**Source {src['source_number']}** &nbsp;|&nbsp; "
                                f"Page **{src['page_number']}** &nbsp;|&nbsp; "
                                f"*{src['section']}*"
                            )
                            st.caption(src["text_preview"])
                            st.divider()

                if msg.get("eval_scores"):
                    sc = msg["eval_scores"]
                    if sc.get("loading"):
                        st.caption("🕒 Running Ragas evaluation in background ...")
                    else:
                        def _p(d): return f"{d.get('score',0):.0%}" if d else "—"
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Context Relevance",  _p(sc.get("context_relevance")))
                        c2.metric("Faithfulness",        _p(sc.get("faithfulness")))
                        c3.metric("Answer Correctness",  _p(sc.get("answer_correctness")))

        st.divider()
        user_query = st.chat_input(
            f"Ask anything about {selected_brand} {selected_model} …"
        )

        if user_query:
            # Dynamically refresh api_key from st.secrets or environment if it is currently empty
            if not st.session_state.api_key:
                try:
                    import os
                    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
                except Exception:
                    pass

            if not st.session_state.vectorstore:
                st.error("No brochure indexed yet. Use the Ingest tab.")
            elif not st.session_state.api_key:
                st.error("Groq API key is missing. Please add GROQ_API_KEY to your Streamlit Secrets or .env file.")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_query}
                )

                with st.spinner("🔗 Running LangGraph pipeline …"):
                    t0 = time.time()

                    # Run pipeline with skip_eval=True for instant answers
                    result = run_rag_pipeline(
                        query=user_query,
                        car_brand=selected_brand,
                        car_model=selected_model,
                        store=st.session_state.vectorstore,
                        api_key=st.session_state.api_key,
                        skip_eval=True,
                    )

                    elapsed = round(time.time() - t0, 2)

                answer  = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                status  = result.get("status", "SUCCESS")
                context = result.get("context", "")

                # Show immediately with loading state for scores
                st.session_state.chat_history.append({
                    "role":        "assistant",
                    "content":     answer,
                    "sources":     sources,
                    "eval_scores": {"loading": True},
                })

                # Log initially to SQLite with NULL scores
                row_id = log_query(
                    selected_brand, selected_model,
                    user_query, answer, sources, elapsed,
                    context_relevance=None,
                    faithfulness=None,
                    answer_correctness=None,
                    status=status,
                )

                # Fire background thread to compute evaluation
                if status == "SUCCESS":
                    ctx = get_script_run_ctx()

                    def eval_bg_worker(q, ctx_text, ans, r_id, api_k, hist_idx):
                        # Re-bind current thread to the Streamlit active context
                        add_script_run_ctx(threading.current_thread(), ctx)
                        
                        from src.evaluator import evaluate_all
                        from src.logger import update_eval_scores
                        import streamlit as st

                        try:
                            # Run evaluation
                            scores = evaluate_all(q, ctx_text, ans, api_k)
                            cr = scores.get("context_relevance", {}).get("score", 0.0)
                            faith = scores.get("faithfulness", {}).get("score", 0.0)
                            ac = scores.get("answer_correctness", {}).get("score", 0.0)
                            
                            # Write final scores to DB
                            update_eval_scores(r_id, cr, faith, ac)
                            
                            # Update local session state if active
                            if hist_idx < len(st.session_state.chat_history):
                                st.session_state.chat_history[hist_idx]["eval_scores"] = scores
                                st.rerun()
                        except Exception as ex:
                            print(f"[AsyncEval] Failed: {ex}")
                            # Clean up loading state if failed
                            if hist_idx < len(st.session_state.chat_history):
                                st.session_state.chat_history[hist_idx]["eval_scores"] = {}
                                st.rerun()

                    thread = threading.Thread(
                        target=eval_bg_worker,
                        args=(user_query, context, answer, row_id, st.session_state.api_key, len(st.session_state.chat_history) - 1)
                    )
                    thread.start()
                else:
                    # Clean up eval scores dict if answer not found
                    st.session_state.chat_history[-1]["eval_scores"] = {}

                st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("<h2 style='color:#e2e8f0'>📄 Upload & Index Car Brochure</h2>", unsafe_allow_html=True)

    left, right = st.columns([2, 1], gap="large")

    with left:
        uploaded = st.file_uploader("Drop your PDF brochure here", type=["pdf"])
        brand    = st.text_input("Car Brand", placeholder="e.g. Hyundai, Maruti, Tata")
        model    = st.text_input("Car Model", placeholder="e.g. i20, Swift, Nexon")
        version  = st.text_input("Year / Version", value="2024")

    with right:
        st.markdown("#### ⚙️ LangChain Pipeline")
        st.markdown("""
        1. 📄 **PyPDFLoader** — page-by-page extraction
        2. ✂️ **RecursiveCharacterTextSplitter** — semantic chunking
        3. 🏷️ Rule-based **section classifier**
        4. 🔢 **HuggingFaceEmbeddings** — local all-MiniLM-L6-v2
        5. 💾 **LangChain FAISS** — vectorstore with metadata
        """)

    st.markdown("")

    if st.button("🚀 Process Brochure", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Please upload a PDF file first.")
        elif not brand.strip() or not model.strip():
            st.error("Please enter both car brand and model.")
        else:
            bar = st.progress(0, text="Saving PDF …")
            try:
                bar.progress(10, text="Saving uploaded PDF …")
                pdf_path = save_uploaded_pdf(uploaded, brand, model)

                bar.progress(25, text="Loading with PyPDFLoader …")
                docs = ingest_brochure(pdf_path, brand, model, version)

                if not docs:
                    st.error("No text could be extracted. Try a different PDF.")
                else:
                    bar.progress(55, text=f"Embedding {len(docs)} chunks with HuggingFaceEmbeddings …")

                    if st.session_state.vectorstore is None:
                        vs = build_vectorstore(docs)
                    else:
                        vs = add_documents_to_store(docs, st.session_state.vectorstore)

                    bar.progress(85, text="Saving LangChain FAISS store …")
                    save_vectorstore(vs)

                    brand_title = brand.strip().title()
                    model_title = model.strip().title()

                    st.session_state.vectorstore   = vs
                    st.session_state.available_cars = get_available_cars(vs)
                    st.session_state.selected_brand = brand_title
                    st.session_state.selected_model = model_title

                    bar.progress(100, text="Done ✅")

                    pages = len({d.metadata.get("page_number") for d in docs})
                    sections = {d.metadata.get("section") for d in docs}

                    # Store results in session state so they persist and are interactive
                    from collections import Counter
                    sec_counts = Counter(d.metadata["section"] for d in docs)
                    sec_df = (
                        pd.DataFrame(sec_counts.items(), columns=["Section", "Chunks"])
                        .sort_values("Chunks", ascending=True)
                    )

                    st.session_state.last_ingested_info = {
                        "brand": brand_title,
                        "model": model_title,
                        "pages": pages,
                        "chunks_count": len(docs),
                        "sections_count": len(sections),
                        "total_vectors": vs.index.ntotal,
                        "sec_df": sec_df,
                        "sections_list": list(sections)
                    }

            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)

    # Render persistent ingestion success state and interactive QA if available
    if st.session_state.last_ingested_info is not None:
        info = st.session_state.last_ingested_info
        st.divider()
        st.success(
            f"✅ **{info['brand']} {info['model']}** successfully indexed!\n\n"
            f"- Pages parsed: **{info['pages']}**\n"
            f"- Chunks indexed: **{info['chunks_count']}**\n"
            f"- Unique sections: **{info['sections_count']}**\n"
            f"- Total store size: **{info['total_vectors']} vectors**\n\n"
            f"👈 **This car has been automatically selected in the sidebar!**\n\n"
            f"You can ask a quick question about it below, or switch to the **💬 Chat Assistant** tab for full thread view."
        )

        col_chart, col_chat = st.columns([1, 1], gap="large")

        with col_chart:
            fig = px.bar(
                info["sec_df"], x="Chunks", y="Section", orientation="h",
                color="Chunks", color_continuous_scale="Viridis",
                title=f"Chunk distribution — {info['brand']} {info['model']}",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_chat:
            st.markdown(f"### 💡 Ask & Get Insights")
            
            # Map sections to suggested questions
            suggestions_map = {
                "Safety Features": "What safety features are available?",
                "Engine & Performance": "What are the engine specs and performance?",
                "Fuel Efficiency": "What is the mileage or fuel efficiency?",
                "Variants & Pricing": "What are the variants and prices?",
                "Comfort & Convenience": "What comfort and convenience features are included?",
                "Infotainment & Connectivity": "What infotainment and connectivity options are there?",
                "Dimensions & Capacity": "What are the dimensions and seating capacity?",
                "Exterior Design": "What are the key exterior design features?",
                "Interior Design": "What are the key interior features?",
                "Warranty & Service": "What is the warranty and service period?",
            }
            
            # Find suggestions based on present sections
            suggestions = []
            for sec in info.get("sections_list", []):
                if sec in suggestions_map:
                    suggestions.append(suggestions_map[sec])
                if len(suggestions) >= 3:
                    break
                    
            # Fallback if not enough suggestions
            fallback_suggestions = [
                f"What are the key specifications of the {info['brand']} {info['model']}?",
                f"Can you summarize the safety and performance features?",
                f"What are the main highlights of this vehicle?"
            ]
            for fs in fallback_suggestions:
                if len(suggestions) >= 3:
                    break
                if fs not in suggestions:
                    suggestions.append(fs)

            # Render suggestion buttons in a clean list
            st.markdown("**💡 Suggested Questions (click to ask):**")
            for idx, sug in enumerate(suggestions):
                if st.button(f"🔍 {sug}", key=f"sug_{idx}", use_container_width=True):
                    st.session_state.ingest_quick_q_input = sug
                    st.rerun()

            st.markdown("")
            
            # Setup a unique key to prevent input collision
            quick_q = st.text_input(
                f"Or ask your own question about **{info['brand']} {info['model']}**:", 
                key="ingest_quick_q_input", 
                placeholder="e.g., What are the safety features? What is the mileage?"
            )
            
            if quick_q:
                if not st.session_state.api_key:
                    st.error("Groq API key is missing. Please add GROQ_API_KEY to your .env file.")
                else:
                    with st.spinner("Retrieving facts and generating insights ..."):
                        ans_res = run_rag_pipeline(
                            query=quick_q,
                            car_brand=info["brand"],
                            car_model=info["model"],
                            store=st.session_state.vectorstore,
                            api_key=st.session_state.api_key,
                            skip_eval=True,
                        )
                        
                    st.markdown("#### 🚗 Insights Response:")
                    st.markdown(
                        f'<div class="bot-bubble">🚗 <b>DriveWise Insights:</b><br><br>{ans_res["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                    
                    if ans_res.get("sources"):
                        with st.expander("📑 View Sources for this answer", expanded=False):
                            for src in ans_res["sources"]:
                                st.markdown(
                                    f"**Source {src['source_number']}** &nbsp;|&nbsp; "
                                    f"Page **{src['page_number']}** &nbsp;|&nbsp; "
                                    f"*{src['section']}*"
                                )
                                st.caption(src["text_preview"])
                                st.divider()
                                
                    # Add a clear button
                    if st.button("🗑️ Clear Answer", key="clear_quick_q", use_container_width=True):
                        st.session_state.ingest_quick_q_input = ""
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("<h2 style='color:#e2e8f0'>📊 Analytics Dashboard</h2>", unsafe_allow_html=True)

    if st.button("🔄 Refresh"):
        st.rerun()

    stats = get_summary_stats()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Queries",      stats["total_queries"])
    k2.metric("Avg Response Time",  f"{stats['avg_response_time']}s")
    k3.metric("Avg Faithfulness",   f"{stats['avg_faithfulness']:.0%}")
    k4.metric("Avg CR",             f"{stats['avg_context_relevance']:.0%}")
    k5.metric("Unanswered",         stats["failed_count"])

    st.divider()
    logs = get_all_logs()

    if not logs:
        st.info("No query logs yet. Start asking questions in the Chat tab!")
    else:
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        cl, cr = st.columns(2)

        with cl:
            st.markdown("#### 📈 Evaluation Scores Over Time")
            edf = df[["timestamp","context_relevance","faithfulness","answer_correctness"]].dropna()
            if not edf.empty:
                fig_ev = go.Figure()
                for col, colour, name in [
                    ("context_relevance",  "#6366f1", "Context Relevance"),
                    ("faithfulness",       "#10b981", "Faithfulness"),
                    ("answer_correctness", "#f59e0b", "Answer Correctness"),
                ]:
                    fig_ev.add_trace(go.Scatter(
                        x=edf["timestamp"], y=edf[col], name=name,
                        line=dict(color=colour, width=2), mode="lines+markers",
                    ))
                fig_ev.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0,1.05], tickformat=".0%"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0,r=0,t=10,b=0),
                )
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.info("Eval scores appear after queries complete.")

        with cr:
            st.markdown("#### ⚡ Response Time Trend")
            tdf = df[["timestamp","response_time_sec"]].dropna()
            if not tdf.empty:
                fig_t = px.line(
                    tdf, x="timestamp", y="response_time_sec",
                    labels={"response_time_sec":"Seconds","timestamp":""},
                    template="plotly_dark",
                )
                fig_t.update_traces(line_color="#f59e0b", line_width=2)
                fig_t.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=10,b=0),
                )
                st.plotly_chart(fig_t, use_container_width=True)

        st.markdown("#### 🚙 Query Distribution by Car")
        cdf = df.groupby(["car_brand","car_model"]).size().reset_index(name="count")
        cdf["car"] = cdf["car_brand"] + " " + cdf["car_model"]
        fig_pie = px.pie(
            cdf, names="car", values="count",
            color_discrete_sequence=px.colors.sequential.Plasma,
            template="plotly_dark",
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

        failed = df[df["status"] == "NO_ANSWER_FOUND"]
        if not failed.empty:
            st.markdown("#### ❌ Unanswered Queries")
            st.dataframe(
                failed[["timestamp","car_brand","car_model","user_query"]].rename(columns={
                    "timestamp":"Time","car_brand":"Brand",
                    "car_model":"Model","user_query":"Question",
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("#### 📋 Full Query Log")
        st.dataframe(
            df[["timestamp","car_brand","car_model","user_query",
                "response_time_sec","context_relevance","faithfulness",
                "answer_correctness","status"]].rename(columns={
                "response_time_sec":"Time(s)","context_relevance":"CR",
                "faithfulness":"Faith","answer_correctness":"AC",
            }),
            use_container_width=True, hide_index=True,
        )
