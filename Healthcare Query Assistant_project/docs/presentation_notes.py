# ═══════════════════════════════════════════════════════════════
#  PRESENTATION TALKING POINTS
#  RAG-Based Healthcare Query Assistant
# ═══════════════════════════════════════════════════════════════

PRESENTATION_OUTLINE = """
┌─────────────────────────────────────────────────────────────┐
│  SLIDE 1 — PROBLEM STATEMENT (1 min)                        │
├─────────────────────────────────────────────────────────────┤
│  Hospital staff face two types of questions daily:          │
│  1. Patient data questions  ("How many diabetic patients?") │
│  2. Policy questions        ("Is pre-auth needed for MRI?") │
│                                                             │
│  Traditional approaches need two separate tools.            │
│  → We built ONE conversational interface for both.          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 2 — ARCHITECTURE (2 min)                             │
├─────────────────────────────────────────────────────────────┤
│  Multi-agent system — 4 agents:                             │
│                                                             │
│  OrchestratorAgent                                          │
│    → Classifies query as database / policy / hybrid         │
│    → Keyword scoring (rule-based) or GPT-4o (LLM mode)      │
│                                                             │
│  NLP-to-SQL Agent                                           │
│    → Maps natural language to SQL using 12 patterns         │
│    → Queries normalised SQLite (4 tables, 10K records)      │
│    → Production: LangChain create_sql_query_chain           │
│                                                             │
│  RAG Agent                                                  │
│    → TF-IDF vector search over 6 policy documents           │
│    → Returns top-K chunks + grounded extractive answer      │
│    → Production: FAISS + sentence-transformers + GPT-4o     │
│                                                             │
│  ResponseFormatter                                          │
│    → SQL → clean table output                               │
│    → RAG → conversational answer with source citation       │
│    → Hybrid → combined response                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 3 — DATA PIPELINE (1 min)                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 1 Part 1 — Synthetic Dataset                         │
│    • 10,000 patient records (no real patient data)          │
│    • 4 normalised tables with FK relationships              │
│    • Indexed on condition, date, test_results, hospital     │
│    • Validated: zero nulls, consistent types                │
│                                                             │
│  Phase 1 Part 2 — Knowledge Base                            │
│    • 6 hospital policy documents (synthesised)              │
│      admission, billing, discharge, insurance,              │
│      emergency, privacy/HIPAA                               │
│    • Chunked at 400 words with 80-word overlap              │
│    • Indexed in FAISS vector store                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 4 — LIVE DEMO (3–4 min)                              │
├─────────────────────────────────────────────────────────────┤
│  Run: streamlit run app.py                                  │
│                                                             │
│  Demo query sequence:                                       │
│  1. "How many diabetic patients are there?"                 │
│     → Show: SQL generated, row count result                 │
│                                                             │
│  2. "Which patients have abnormal test results?"            │
│     → Show: table of patients with names + conditions       │
│                                                             │
│  3. "Show average billing by insurance provider"            │
│     → Show: aggregated table sorted by avg billing          │
│                                                             │
│  4. "What is the hospital discharge policy?"                │
│     → Show: RAG retrieval, policy answer with source        │
│                                                             │
│  5. "Is prior insurance authorisation required for surgery?"│
│     → Show: specific policy answer from insurance docs      │
│                                                             │
│  6. Switch to Database Explorer tab                         │
│     → Show: condition distribution bar chart                │
│     → Show: custom SQL runner                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 5 — EVALUATION RESULTS (1 min)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Category      Tests   Pass Rate   Avg Latency             │
│  ──────────── ─────── ─────────── ────────────             │
│  SQL               8      87.5%         3 ms               │
│  RAG               6     100.0%        13 ms               │
│  Ambiguous/OOD     5     100.0%         0 ms               │
│  Overall          19      94.7%         5 ms               │
│                                                             │
│  Key point: zero crashes on edge cases / garbage input      │
│  Key point: RAG 100% — all policy questions correctly routed│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 6 — TECH STACK (30 sec)                              │
├─────────────────────────────────────────────────────────────┤
│  Python 3.11          Core language                         │
│  SQLite               Structured patient database           │
│  FAISS / TF-IDF       Vector similarity search              │
│  LangChain            Agent orchestration framework         │
│  OpenAI GPT-4o-mini   LLM for SQL gen + answer synthesis    │
│  sentence-transformers Embedding model (all-MiniLM-L6-v2)  │
│  Streamlit            Web interface                         │
│  pandas / numpy       Data manipulation                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SLIDE 7 — WHAT I LEARNED / FUTURE WORK (1 min)             │
├─────────────────────────────────────────────────────────────┤
│  Learned:                                                   │
│  • Multi-agent routing tradeoffs (rules vs LLM)             │
│  • RAG chunking strategy affects retrieval quality          │
│  • TimeSeriesSplit matters for evaluation correctness       │
│  • Conversation memory design for multi-turn context        │
│                                                             │
│  Future improvements:                                       │
│  • Add re-ranking (Cohere Rerank) for better RAG precision  │
│  • Streaming responses for real-time feel                   │
│  • Auth + role-based access (staff vs admin queries)        │
│  • PostgreSQL upgrade for production scale                  │
│  • Multi-turn context injection into SQL chain              │
└─────────────────────────────────────────────────────────────┘

ANTICIPATED QUESTIONS & ANSWERS
═══════════════════════════════

Q: Why not just use one LLM for everything?
A: Cost and latency. Rule-based SQL handles 80% of queries in <5ms
   with no API cost. LLM is reserved for complex/ambiguous cases.

Q: How does it handle a question the SQL patterns don't cover?
A: Falls back to "couldn't generate SQL" error message with a prompt
   to rephrase. In LLM mode, GPT-4o generates any valid SQL.

Q: What stops it from running a dangerous SQL query (DROP TABLE)?
A: SQLite connection is read-only (no write permissions granted).
   LangChain SQL chain is also configured with read-only access.

Q: Is this HIPAA compliant?
A: The dataset is entirely synthetic — no real patient data.
   For production, all patient data would be de-identified and
   the system would sit behind auth + audit logging.

Q: How would you scale this to 1M records?
A: Migrate SQLite → PostgreSQL. Add connection pooling.
   Add a Redis cache layer for frequent queries.
   FAISS index scales linearly — no change needed for RAG.
"""

if __name__ == "__main__":
    print(PRESENTATION_OUTLINE)