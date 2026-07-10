"""
app.py — Streamlit Web Interface
RAG-Based Healthcare Query Assistant
Run: streamlit run app.py
"""

import os
import sys
import time
import json
import sqlite3
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
  .intent-badge {
      display: inline-block; padding: 2px 10px; border-radius: 99px;
      font-size: 12px; font-weight: 600; margin-left: 8px;
  }
  .badge-db     { background:#dbeafe; color:#1d4ed8; }
  .badge-policy { background:#dcfce7; color:#15803d; }
  .badge-hybrid { background:#fef9c3; color:#854d0e; }
  .metric-box {
      background: #f8fafc; border-radius: 10px;
      padding: 14px 18px; border: 1px solid #e2e8f0;
  }
  .route-log {
      font-family: monospace; font-size: 12px;
      background:#f1f5f9; padding:8px 12px; border-radius:8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages"      : [],
        "routing_log"   : [],
        "agents_ready"  : False,
        "total_queries" : 0,
        "sql_count"     : 0,
        "rag_count"     : 0,
        "hybrid_count"  : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load agents (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agents…")
def load_agents():
    from data.generate_dataset import generate_patients, build_database
    from knowledge_base.policy_generator import build_knowledge_base
    from agents.agents import (OrchestratorAgent, NLPtoSQLAgent,
                                RAGAgent, ResponseFormatter)

    db_path    = os.path.join(ROOT, "database/healthcare.db")
    index_path = os.path.join(ROOT, "knowledge_base/faiss_index")

    if not os.path.exists(db_path):
        df = generate_patients()
        build_database(df, db_path)

    if not os.path.exists(os.path.join(index_path, "store.pkl")):
        build_knowledge_base()

    return {
        "orchestrator" : OrchestratorAgent(),
        "sql"          : NLPtoSQLAgent(db_path=db_path),
        "rag"          : RAGAgent(index_path=index_path),
        "formatter"    : ResponseFormatter(),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=64)
    st.title("Healthcare Assistant")
    st.caption("Multi-Agent RAG System")
    st.divider()

    # Live metrics
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Total Queries",  st.session_state.total_queries)
    col2.metric("SQL Queries",    st.session_state.sql_count)
    col1.metric("Policy Queries", st.session_state.rag_count)
    col2.metric("Hybrid",         st.session_state.hybrid_count)

    st.divider()

    # Sample queries
    st.subheader("💡 Sample Queries")
    SAMPLES = {
        "🗄️ Patient Data": [
            "How many diabetic patients are there?",
            "Which patients have abnormal test results?",
            "Show average billing by insurance provider",
            "How many emergency admissions are there?",
            "What is the most common medical condition?",
            "Show gender distribution of patients",
        ],
        "📋 Hospital Policy": [
            "What is the hospital discharge policy?",
            "Is prior insurance authorisation required for surgery?",
            "What is the emergency triage protocol?",
            "What are patient rights under HIPAA?",
            "How does the billing dispute process work?",
        ],
    }
    for category, queries in SAMPLES.items():
        with st.expander(category):
            for q in queries:
                if st.button(q, key=f"btn_{q[:30]}", use_container_width=True):
                    st.session_state["prefill"] = q

    st.divider()

    # Agent routing log
    if st.session_state.routing_log:
        st.subheader("🔀 Routing Log")
        for entry in st.session_state.routing_log[-5:]:
            badge_cls = {"database":"badge-db","policy":"badge-policy"}.get(
                entry["intent"], "badge-hybrid")
            st.markdown(
                f'<div class="route-log">'
                f'<span class="intent-badge {badge_cls}">{entry["intent"].upper()}</span> '
                f'{entry["query"][:40]}… <br>'
                f'<small>{entry["latency_ms"]:.0f}ms</small></div>',
                unsafe_allow_html=True,
            )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.routing_log   = []
        st.session_state.total_queries = 0
        st.session_state.sql_count     = 0
        st.session_state.rag_count     = 0
        st.session_state.hybrid_count  = 0
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏥 Healthcare Query Assistant")
st.caption("Ask about patient records (SQL) or hospital policies (RAG) in plain English.")

# Tabs
tab_chat, tab_db, tab_eval = st.tabs(["💬 Chat", "🗄️ Database Explorer", "📈 Evaluation"])

# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta   = msg["meta"]
                intent = meta.get("intent", "")
                badge  = {"database":"🗄️ SQL","policy":"📋 Policy","hybrid":"🔀 Hybrid"}.get(intent,"❓")
                st.caption(f"{badge} | {meta.get('latency_ms',0):.0f} ms")

    # Prefill from sidebar button click
    prefill = st.session_state.pop("prefill", None)

    # Chat input
    user_input = st.chat_input("Ask about patients or hospital policies…")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agents
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    agents = load_agents()
                    t0     = time.time()
                    intent = agents["orchestrator"].classify_intent(user_input)
                    resp   = agents["orchestrator"].route(
                        user_input,
                        agents["sql"],
                        agents["rag"],
                        agents["formatter"],
                    )
                    ms = (time.time() - t0) * 1000
                except Exception as e:
                    resp   = f"⚠️ Error: {e}"
                    intent = "error"
                    ms     = 0

            st.markdown(resp)
            badge = {"database":"🗄️ SQL","policy":"📋 Policy","hybrid":"🔀 Hybrid"}.get(intent,"❓")
            st.caption(f"{badge} | {ms:.0f} ms")

        # Update state
        st.session_state.messages.append({
            "role": "assistant", "content": resp,
            "meta": {"intent": intent, "latency_ms": ms},
        })
        st.session_state.routing_log.append({
            "query": user_input, "intent": intent, "latency_ms": ms
        })
        st.session_state.total_queries += 1
        if intent == "database": st.session_state.sql_count    += 1
        elif intent == "policy": st.session_state.rag_count    += 1
        elif intent == "hybrid": st.session_state.hybrid_count += 1

        st.rerun()

# ── TAB 2: Database Explorer ──────────────────────────────────────────────────
with tab_db:
    st.subheader("🗄️ Database Explorer")
    db_path = os.path.join(ROOT, "database/healthcare.db")

    if not os.path.exists(db_path):
        st.warning("Database not found. Run `python main.py --setup` first.")
    else:
        conn = sqlite3.connect(db_path)

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients",    pd.read_sql("SELECT COUNT(*) as n FROM patients",      conn)["n"][0])
        c2.metric("Abnormal Results",  pd.read_sql("SELECT COUNT(*) as n FROM clinical_data WHERE test_results='Abnormal'", conn)["n"][0])
        c3.metric("Emergency Admissions", pd.read_sql("SELECT COUNT(*) as n FROM admissions WHERE admission_type='Emergency'", conn)["n"][0])
        c4.metric("Avg Billing ($)",   f"{pd.read_sql('SELECT AVG(billing_amount) as n FROM billing', conn)['n'][0]:,.0f}")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Conditions Distribution**")
            df_cond = pd.read_sql(
                "SELECT medical_condition, COUNT(*) as count FROM clinical_data "
                "GROUP BY medical_condition ORDER BY count DESC", conn)
            st.bar_chart(df_cond.set_index("medical_condition"))

        with col_b:
            st.markdown("**Avg Billing by Insurance**")
            df_bill = pd.read_sql(
                "SELECT insurance_provider, ROUND(AVG(billing_amount),0) as avg_billing "
                "FROM billing GROUP BY insurance_provider ORDER BY avg_billing DESC", conn)
            st.bar_chart(df_bill.set_index("insurance_provider"))

        st.divider()
        st.markdown("**Run Custom SQL**")
        custom_sql = st.text_area("SQL Query", value="SELECT p.name, p.age, c.medical_condition, b.billing_amount\nFROM patients p\nJOIN clinical_data c ON p.patient_id = c.patient_id\nJOIN billing b ON p.patient_id = b.patient_id\nLIMIT 10", height=100)
        if st.button("▶ Run Query"):
            try:
                result_df = pd.read_sql(custom_sql, conn)
                st.dataframe(result_df, use_container_width=True)
                st.caption(f"{len(result_df)} rows returned")
            except Exception as e:
                st.error(f"SQL Error: {e}")

        conn.close()

# ── TAB 3: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📈 Evaluation Dashboard")

    if st.button("▶ Run Full Evaluation Suite", type="primary"):
        with st.spinner("Running 19 test cases…"):
            import subprocess, json as _json
            result = subprocess.run(
                [sys.executable, "tests/evaluate.py"],
                capture_output=True, text=True, cwd=ROOT
            )
            st.code(result.stdout, language="text")

    # Load latest eval report if exists
    log_dir = os.path.join(ROOT, "logs")
    reports = sorted([f for f in os.listdir(log_dir) if f.startswith("eval_report")], reverse=True)
    if reports:
        with open(os.path.join(log_dir, reports[0])) as f:
            data = json.load(f)

        df_eval = pd.DataFrame([{
            "ID"        : r["id"],
            "Category"  : r["category"],
            "Description": r["description"],
            "Passed"    : "✅" if r["passed"] else "❌",
            "Latency ms": r["latency_ms"],
        } for r in data])

        total  = len(data)
        passed = sum(1 for r in data if r["passed"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Accuracy", f"{passed/total*100:.1f}%", f"{passed}/{total} tests")
        m2.metric("SQL Accuracy",     f"{sum(1 for r in data if r['category']=='SQL' and r['passed'])}/{sum(1 for r in data if r['category']=='SQL')*1:.0f}" )
        m3.metric("RAG Relevance",    "100%" if all(r["passed"] for r in data if r["category"]=="RAG") else "Partial")

        st.divider()
        st.dataframe(df_eval, use_container_width=True, hide_index=True)