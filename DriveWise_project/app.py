"""
app.py — DriveWise Streamlit Web Interface
Run: streamlit run app.py
"""

import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="DriveWise — Car Brochure Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .brand-badge {
    display:inline-block; padding:3px 12px; border-radius:99px;
    font-size:12px; font-weight:600; background:#dbeafe; color:#1d4ed8;
  }
  .source-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
    padding:10px 14px; margin:4px 0; font-size:13px;
  }
  .metric-highlight { font-size:22px; font-weight:600; color:#1d4ed8; }
  .score-good  { color: #15803d; font-weight:600; }
  .score-mid   { color: #b45309; font-weight:600; }
  .score-low   { color: #b91c1c; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init():
    defaults = {
        "messages"   : [],
        "brand"      : None,
        "model"      : None,
        "rag"        : None,
        "last_sources": [],
        "total_q"    : 0,
        "avg_lat"    : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()


# ── Load RAG (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DriveWise engine…")
def load_rag():
    from Core.rag_engine import DriveWiseRAG, build_vector_store, DriveWiseVectorStore
    from Core.logger     import QueryLogger, RAGEvaluator

    store_path = os.path.join(ROOT, "vector_store", "drivewise_store.pkl")
    if not os.path.exists(store_path):
        build_vector_store()
    store = DriveWiseVectorStore.load(store_path)
    return DriveWiseRAG(store), QueryLogger(), RAGEvaluator()


rag_engine, qlogger, evaluator = load_rag()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/car.png", width=56)
    st.title("DriveWise")
    st.caption("Brochure-Grounded AI Assistant")
    st.divider()

    # Brand selector
    brands = rag_engine.get_brands()
    brand  = st.selectbox("Select Brand", ["— Choose —"] + brands)
    if brand and brand != "— Choose —":
        models = rag_engine.get_models(brand)
        model  = st.selectbox("Select Model", ["— Choose —"] + models)
        if model and model != "— Choose —":
            if brand != st.session_state.brand or model != st.session_state.model:
                st.session_state.brand    = brand
                st.session_state.model    = model
                st.session_state.messages = []
                st.session_state.last_sources = []

    st.divider()

    # Sample questions
    st.subheader("💡 Sample Questions")
    SAMPLES = [
        "What is the mileage of the petrol variant?",
        "How many airbags does it have?",
        "What is the boot space?",
        "What is the price of the top variant?",
        "Does it have a sunroof?",
        "What is the NCAP safety rating?",
        "What is the ground clearance?",
        "Is AWD / 4WD available?",
        "What touchscreen size does it have?",
        "Which variant has ADAS features?",
    ]
    for q in SAMPLES:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()

    # Session stats
    st.subheader("📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Queries",    st.session_state.total_q)
    c2.metric("Avg ms",     f"{st.session_state.avg_lat:.0f}")

    # Last sources
    if st.session_state.last_sources:
        st.subheader("📄 Last Sources")
        for s in st.session_state.last_sources:
            st.markdown(
                f'<div class="source-card">📘 <b>{s["section"]}</b><br>'
                f'<small>Page {s["page_number"]} · Score {s["score"]:.3f}</small></div>',
                unsafe_allow_html=True
            )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.last_sources  = []
        st.session_state.total_q       = 0
        st.session_state.avg_lat       = 0
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🚗 DriveWise")
st.caption("Metadata-Aware Automotive RAG Assistant · Ask anything from official brochures")

tab_chat, tab_compare, tab_eval = st.tabs(["💬 Chat", "🔍 Compare Specs", "📈 Evaluation"])

# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.brand or not st.session_state.model:
        st.info("👈 Select a brand and model from the sidebar to start asking questions.")
    else:
        st.markdown(
            f'<span class="brand-badge">🚗 {st.session_state.brand} {st.session_state.model}</span>',
            unsafe_allow_html=True
        )
        st.caption("Answers grounded in official brochure data only.")

        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    m = msg["meta"]
                    score_cls = "score-good" if m.get("faith",0) > 0.6 else "score-mid"
                    st.caption(
                        f"⏱ {m.get('latency_ms',0):.0f}ms · "
                        f"Faithfulness: {m.get('faith',0):.0%} · "
                        f"Relevance: {m.get('rel',0):.0%}"
                    )

        # Input
        prefill    = st.session_state.pop("prefill", None)
        user_input = st.chat_input("Ask about the selected car…")
        if prefill and not user_input:
            user_input = prefill

        if user_input:
            if not st.session_state.brand or st.session_state.brand == "— Choose —":
                st.warning("Please select a brand and model first.")
            else:
                st.session_state.messages.append({"role":"user","content":user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Searching brochure…"):
                        result  = rag_engine.query(
                            user_input,
                            st.session_state.brand,
                            st.session_state.model
                        )
                        qlogger.log(result)
                        metrics = evaluator.evaluate(user_input, result["answer"], [])

                    st.markdown(result["answer"])

                    # Source cards inline
                    if result["sources"]:
                        with st.expander("📄 Sources", expanded=False):
                            for s in result["sources"]:
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'📘 <b>{s["section"]}</b> · Page {s["page_number"]}'
                                    f'<br><small>{s["doc_name"]}</small></div>',
                                    unsafe_allow_html=True
                                )

                    faith = metrics["faithfulness"]
                    rel   = metrics["context_relevance"]
                    st.caption(
                        f"⏱ {result['latency_ms']:.0f}ms · "
                        f"Faithfulness: {faith:.0%} · Relevance: {rel:.0%}"
                    )

                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "meta"   : {"latency_ms": result["latency_ms"],
                                "faith": metrics["faithfulness"],
                                "rel"  : metrics["context_relevance"]},
                })
                st.session_state.last_sources = result["sources"]

                # Update stats
                n = st.session_state.total_q
                st.session_state.avg_lat = (
                    (st.session_state.avg_lat * n + result["latency_ms"]) / (n + 1)
                )
                st.session_state.total_q += 1
                st.rerun()


# ── TAB 2: Compare Specs ──────────────────────────────────────────────────────
with tab_compare:
    st.subheader("🔍 Compare Two Cars on Any Spec")
    col1, col2 = st.columns(2)

    with col1:
        b1 = st.selectbox("Brand 1", ["— Choose —"] + rag_engine.get_brands(), key="b1")
        if b1 and b1 != "— Choose —":
            m1 = st.selectbox("Model 1", ["— Choose —"] + rag_engine.get_models(b1), key="m1")

    with col2:
        b2 = st.selectbox("Brand 2", ["— Choose —"] + rag_engine.get_brands(), key="b2")
        if b2 and b2 != "— Choose —":
            m2 = st.selectbox("Model 2", ["— Choose —"] + rag_engine.get_models(b2), key="m2")

    spec_q = st.text_input(
        "Spec to compare",
        placeholder="e.g. What is the mileage? / How many airbags? / What is the price?"
    )

    if st.button("⚡ Compare", type="primary"):
        if (b1 and b1 != "— Choose —" and "m1" in st.session_state and
            st.session_state.m1 != "— Choose —" and
            b2 and b2 != "— Choose —" and "m2" in st.session_state and
            st.session_state.m2 != "— Choose —" and spec_q):

            c1, c2 = st.columns(2)
            r1 = rag_engine.query(spec_q, b1, st.session_state.m1)
            r2 = rag_engine.query(spec_q, b2, st.session_state.m2)

            with c1:
                st.markdown(f"### {b1} {st.session_state.m1}")
                st.markdown(r1["answer"])
                for s in r1["sources"]:
                    st.caption(f"📘 {s['section']} · Page {s['page_number']}")

            with c2:
                st.markdown(f"### {b2} {st.session_state.m2}")
                st.markdown(r2["answer"])
                for s in r2["sources"]:
                    st.caption(f"📘 {s['section']} · Page {s['page_number']}")
        else:
            st.warning("Please select both cars and enter a spec question.")


# ── TAB 3: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📈 Evaluation Dashboard")
    st.caption("Runs 20 test questions and measures answer quality.")

    if st.button("▶ Run Full Evaluation (20 tests)", type="primary"):
        import subprocess
        with st.spinner("Running evaluation suite… (~30 seconds)"):
            r = subprocess.run(
                [sys.executable, "tests/evaluate.py"],
                capture_output=True, text=True, cwd=ROOT
            )
        st.code(r.stdout, language="text")
        if r.stderr:
            st.code(r.stderr[:500], language="text")

    # Show latest report
    log_dir = os.path.join(ROOT, "logs")
    if os.path.exists(log_dir):
        reports = sorted(
            [f for f in os.listdir(log_dir) if f.startswith("eval_report")],
            reverse=True
        )
        if reports:
            with open(os.path.join(log_dir, reports[0])) as f:
                data = json.load(f)

            total  = len(data)
            passed = sum(1 for r in data if r.get("passed"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pass Rate",     f"{passed}/{total}")
            m2.metric("Avg Correctness", f"{sum(r['answer_correctness'] for r in data)/total:.0%}")
            m3.metric("Avg Faithfulness",f"{sum(r['faithfulness'] for r in data)/total:.0%}")
            m4.metric("Avg Latency",   f"{sum(r['latency_ms'] for r in data)/total:.0f}ms")

            st.divider()
            import pandas as pd
            df = pd.DataFrame([{
                "Brand"       : r["brand"],
                "Model"       : r["model"],
                "Question"    : r["question"][:55],
                "Passed"      : "✅" if r["passed"] else "❌",
                "Correctness" : f"{r['answer_correctness']:.0%}",
                "Faithfulness": f"{r['faithfulness']:.0%}",
                "Relevance"   : f"{r['context_relevance']:.0%}",
                "Latency ms"  : r["latency_ms"],
            } for r in data])
            st.dataframe(df, use_container_width=True, hide_index=True)