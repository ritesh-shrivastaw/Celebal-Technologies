"""
Phase 1 – Part 3
Multi-Agent System:
  OrchestratorAgent  → detects intent, routes to correct agent
  NLPtoSQLAgent      → converts natural language to SQL, queries SQLite
  RAGAgent           → retrieves policy chunks, generates grounded answers
  ResponseFormatter  → produces clean conversational output
"""

import os
import re
import json
import sqlite3
import logging
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), "../logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "../logs/agent_routing.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DB_PATH    = os.path.join(os.path.dirname(__file__), "../database/healthcare.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/faiss_index")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
class OrchestratorAgent:
    """
    Analyses query intent and routes to the appropriate agent.
    Intent classes:
      - 'database'  → NLPtoSQLAgent  (patient data, stats, counts)
      - 'policy'    → RAGAgent        (hospital policies, procedures, insurance)
      - 'hybrid'    → both agents, results merged
      - 'unknown'   → fallback response
    """

    # Keywords that strongly indicate a DATABASE query
    DB_KEYWORDS = [
        "patient", "patients", "admitted", "doctor",
        "medication", "diagnos", "condition", "blood type", "test result",
        "age", "gender", "how many", "count", "average", "total", "sum",
        "list", "show", "which patient", "record", "room", "nurse",
    ]

    # Keywords that strongly indicate a POLICY query
    POLICY_KEYWORDS = [
        "policy", "procedure", "protocol", "guideline", "rule", "regulation",
        "authorisation", "authorization", "pre-auth", "prior approval",
        "discharge policy", "discharge process", "how to", "what is the",
        "what are the requirements", "emergency protocol", "triage",
        "hipaa", "privacy", "consent", "financial hardship", "charity care",
        "appeal", "dispute", "required for", "needed for", "qualify", "eligible",
        "rights", "billing dispute", "insurance policy", "insurance requirement",
    ]

    def classify_intent(self, query: str) -> str:
        q = query.lower()

        # Strong policy signals override everything
        strong_policy = any(kw in q for kw in [
            "policy", "protocol", "guideline", "authorisation", "authorization",
            "hipaa", "privacy", "triage", "appeal", "dispute", "required for",
            "rights", "procedure", "regulation",
        ])
        if strong_policy:
            logging.info(f"INTENT | query='{query[:80]}' | strong_policy → policy")
            return "policy"

        db_score     = sum(1 for kw in self.DB_KEYWORDS     if kw in q)
        policy_score = sum(1 for kw in self.POLICY_KEYWORDS if kw in q)

        if db_score > 0 and policy_score > 0:
            intent = "hybrid"
        elif db_score >= policy_score and db_score > 0:
            intent = "database"
        elif policy_score > db_score:
            intent = "policy"
        else:
            intent = "unknown"

        logging.info(f"INTENT | query='{query[:80]}' | db={db_score} | policy={policy_score} | → {intent}")
        return intent

    def route(self, query: str, sql_agent, rag_agent, formatter) -> str:
        intent  = self.classify_intent(query)
        routing = {"query": query, "intent": intent, "timestamp": datetime.now().isoformat()}

        if intent == "database":
            result = sql_agent.run(query)
            routing["agent_used"] = "NLPtoSQLAgent"

        elif intent == "policy":
            result = rag_agent.run(query)
            routing["agent_used"] = "RAGAgent"

        elif intent == "hybrid":
            db_result  = sql_agent.run(query)
            rag_result = rag_agent.run(query)
            result = {"type": "hybrid", "database": db_result, "policy": rag_result}
            routing["agent_used"] = "Both"

        else:
            result = {
                "type"   : "fallback",
                "message": ("I can help with patient data queries (e.g. admissions, billing) "
                            "or hospital policy questions. Could you rephrase your question?")
            }
            routing["agent_used"] = "Fallback"

        logging.info(f"ROUTING | {json.dumps(routing)}")
        return formatter.format(result, intent, query)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — NLP-TO-SQL
# ═══════════════════════════════════════════════════════════════════════════════
class NLPtoSQLAgent:
    """
    Converts natural language questions into SQL and executes them
    against the SQLite healthcare database.

    Uses rule-based pattern matching here; plug in LangChain SQL chain
    or an LLM call for production (see _llm_generate_sql below).
    """

    SCHEMA = """
    Tables:
      patients(patient_id, name, age, gender, blood_type)
      clinical_data(record_id, patient_id, medical_condition, medication, test_results)
      admissions(admission_id, patient_id, hospital, doctor, room_number,
                 admission_type, admission_date, discharge_date)
      billing(billing_id, patient_id, insurance_provider, billing_amount)

    Relationships: patients ← clinical_data, admissions, billing (patient_id FK)
    """

    # ── Pattern → SQL template map ─────────────────────────────────────────────
    # NOTE: Order matters — more specific patterns first, generic last.
    PATTERNS = [
        # Abnormal test results  (specific — before generic patient patterns)
        (r"abnormal test",
         "SELECT p.name, p.age, c.medical_condition, c.test_results "
         "FROM patients p JOIN clinical_data c ON p.patient_id = c.patient_id "
         "WHERE c.test_results = 'Abnormal' LIMIT 20"),

        # Average billing by insurance
        (r"average billing|billing amount.*(by|per|for each) insurance",
         "SELECT b.insurance_provider, ROUND(AVG(b.billing_amount),2) as avg_billing, "
         "COUNT(*) as patient_count "
         "FROM billing b GROUP BY b.insurance_provider ORDER BY avg_billing DESC"),

        # Total billing / revenue
        (r"total billing|total revenue",
         "SELECT ROUND(SUM(billing_amount),2) as total_billing FROM billing"),

        # Admitted last month
        (r"admitted last month",
         "SELECT p.name, a.admission_date, a.hospital, c.medical_condition "
         "FROM patients p "
         "JOIN admissions a ON p.patient_id = a.patient_id "
         "JOIN clinical_data c ON p.patient_id = c.patient_id "
         "WHERE strftime('%Y-%m', a.admission_date) = "
         "strftime('%Y-%m', date('now', '-1 month')) LIMIT 20"),

        # Count by hospital
        (r"patients.*(at|in) each hospital|each hospital|by hospital|per hospital",
         "SELECT hospital, COUNT(*) as patient_count "
         "FROM admissions GROUP BY hospital ORDER BY patient_count DESC"),

        # Most common / top conditions
        (r"most common (medical )?condition|top condition|frequent condition",
         "SELECT medical_condition, COUNT(*) as count "
         "FROM clinical_data GROUP BY medical_condition ORDER BY count DESC LIMIT 10"),

        # Emergency admissions
        (r"emergency admission",
         "SELECT COUNT(*) as emergency_count FROM admissions "
         "WHERE admission_type = 'Emergency'"),

        # Average age
        (r"average age",
         "SELECT ROUND(AVG(age),1) as average_age FROM patients"),

        # Gender distribution
        (r"gender (distribution|breakdown|split)|show gender|male.*female|female.*male",
         "SELECT gender, COUNT(*) as count FROM patients GROUP BY gender"),

        # Patients by doctor
        (r"patients (under|of|for|assigned to) dr|dr\.?\s?(\w+)'s patients",
         "SELECT p.name, p.age, c.medical_condition, a.admission_date "
         "FROM patients p "
         "JOIN admissions a ON p.patient_id = a.patient_id "
         "JOIN clinical_data c ON p.patient_id = c.patient_id "
         "WHERE LOWER(a.doctor) LIKE LOWER('%{0}%') LIMIT 15"),

        # Count diabetic / condition-specific patients  (how many X patients)
        (r"how many (\w+(?:\s+\w+)?) patients|how many patients with (\w+(?:\s+\w+)?)",
         "SELECT COUNT(*) as count FROM patients p "
         "JOIN clinical_data c ON p.patient_id = c.patient_id "
         "WHERE LOWER(c.medical_condition) LIKE LOWER('%{0}%')"),

        # Patients with a specific condition  (list them)
        (r"patients with (\w+(?:\s+\w+)?)",
         "SELECT p.name, p.age, p.gender, c.medical_condition, c.test_results "
         "FROM patients p JOIN clinical_data c ON p.patient_id = c.patient_id "
         "WHERE LOWER(c.medical_condition) LIKE LOWER('%{0}%') LIMIT 15"),
    ]

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _extract_group(self, pattern: str, query: str) -> str:
        """Extract first non-None group from regex match."""
        m = re.search(pattern, query, re.IGNORECASE)
        if not m:
            return ""
        groups = [g for g in m.groups() if g is not None]
        return groups[0].strip() if groups else ""

    def generate_sql(self, query: str) -> tuple[str, dict]:
        """
        Rule-based NL → SQL.
        In production, replace with:
          from langchain.chains import create_sql_query_chain
          from langchain_openai import ChatOpenAI
          llm   = ChatOpenAI(model="gpt-4o-mini")
          chain = create_sql_query_chain(llm, db)
          sql   = chain.invoke({"question": query})
        """
        q = query.lower()
        for pattern, sql_template in self.PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                param = self._extract_group(pattern, query)
                sql   = sql_template.format(param) if "{0}" in sql_template else sql_template
                return sql, {"pattern_matched": pattern}
        return None, {}

    def _llm_generate_sql(self, query: str) -> str:
        """
        PRODUCTION: Use this with LangChain + OpenAI.
        Uncomment when API key is available.

        from langchain_community.utilities import SQLDatabase
        from langchain.chains import create_sql_query_chain
        from langchain_openai import ChatOpenAI

        db    = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = create_sql_query_chain(llm, db)
        return chain.invoke({"question": query})
        """
        raise NotImplementedError("Set OPENAI_API_KEY and uncomment LangChain code")

    _SYNONYMS = {
        "diabetic":"diabetes","hypertensive":"hypertension","asthmatic":"asthma",
        "arthritic":"arthritis","cancerous":"cancer","cardiac":"heart disease",
        "obese":"obesity","anxious":"anxiety","depressed":"depression",
        "covid":"covid-19","anemic":"anemia","anaemic":"anemia",
        "renal":"kidney disease",
    }

    def _normalise(self, q: str) -> str:
        ql = q.lower()
        for adj, noun in self._SYNONYMS.items():
            if adj in ql:
                ql = ql.replace(adj, noun)
        return ql

    def run(self, query: str) -> dict:
        query = self._normalise(query)
        sql, meta = self.generate_sql(query)
        if not sql:
            return {"type": "sql_error", "message": "Could not generate SQL for this query.",
                    "query": query}
        try:
            conn    = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor  = conn.cursor()
            cursor.execute(sql)
            rows    = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()
            data = [dict(zip(columns, row)) for row in rows]
            logging.info(f"SQL | query='{query[:60]}' | sql='{sql[:80]}' | rows={len(data)}")
            return {"type": "sql_result", "sql": sql, "data": data,
                    "row_count": len(data), "columns": columns}
        except Exception as e:
            logging.error(f"SQL ERROR | {e} | sql={sql}")
            return {"type": "sql_error", "message": str(e), "sql": sql}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — RAG AGENT
# ═══════════════════════════════════════════════════════════════════════════════
class RAGAgent:
    """
    Retrieves relevant policy document chunks via vector search,
    then generates a grounded answer.

    Retrieval: SimpleVectorStore (TF-IDF) — swap for FAISS + sentence-transformers.
    Generation: Template-based — swap for LLM with retrieved context.
    """

    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path
        self._store     = None

    def _load_store(self):
        if self._store is None:
            import sys, importlib.util
            kb_path = os.path.join(os.path.dirname(__file__), "../knowledge_base/policy_generator.py")
            spec = importlib.util.spec_from_file_location("policy_generator", kb_path)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules["policy_generator"] = mod
            spec.loader.exec_module(mod)
            self._store = mod.SimpleVectorStore.load(self.index_path)

    def _llm_generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """
        PRODUCTION: Replace with LangChain RAG chain.

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = ChatPromptTemplate.from_template(
            "You are a hospital policy assistant. Answer the question using ONLY "
            "the provided context. If the answer is not in context, say so.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        chain  = prompt | llm | StrOutputParser()
        return chain.invoke({"context": '\n---\n'.join(context_chunks), "question": query})
        """
        # Fallback: return top chunk with framing
        return None  # triggers template-based answer below

    def run(self, query: str, top_k: int = 4) -> dict:
        self._load_store()
        results = self._store.search(query, top_k=top_k)
        if not results:
            return {"type": "rag_result", "answer": "No relevant policy documents found.",
                    "sources": [], "chunks": []}

        top_chunks  = [r["chunk"]  for r in results]
        sources     = list({r["meta"]["source"] for r in results})

        # Try LLM generation; fall back to extractive answer
        answer = self._llm_generate_answer(query, top_chunks)
        if answer is None:
            # Extractive: pick most relevant sentences from top chunk
            sentences = re.split(r"(?<=[.!?])\s+", top_chunks[0])
            answer    = " ".join(sentences[:5])

        logging.info(f"RAG | query='{query[:60]}' | sources={sources} | chunks={len(results)}")
        return {"type": "rag_result", "answer": answer,
                "sources": sources, "chunks": top_chunks, "scores": [r["score"] for r in results]}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — RESPONSE FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════
class ResponseFormatter:
    """Converts raw agent output into clean, conversational responses."""

    def format(self, result: dict, intent: str, original_query: str) -> str:
        if intent == "database":
            return self._format_sql(result, original_query)
        elif intent == "policy":
            return self._format_rag(result)
        elif intent == "hybrid":
            sql_part = self._format_sql(result.get("database", {}), original_query)
            rag_part = self._format_rag(result.get("policy", {}))
            return f"{sql_part}\n\n{'─'*50}\n\n**Related Policy:**\n{rag_part}"
        else:
            return result.get("message", "I'm sorry, I couldn't understand that query.")

    def _format_sql(self, result: dict, query: str) -> str:
        if result.get("type") == "sql_error":
            return f"⚠️  I couldn't retrieve that data: {result.get('message')}"

        data      = result.get("data", [])
        row_count = result.get("row_count", 0)

        if not data:
            return "No records found matching your query."

        # Single scalar result (COUNT, AVG, SUM)
        if row_count == 1 and len(result.get("columns", [])) == 1:
            col, val = result["columns"][0], list(data[0].values())[0]
            return f"**Result:** {val:,}" if isinstance(val, (int, float)) else f"**Result:** {val}"

        # Single-row single aggregation
        if row_count == 1:
            row   = data[0]
            lines = [f"  • **{k.replace('_',' ').title()}**: {v}" for k, v in row.items()]
            return "\n".join(lines)

        # Multi-row: table-style
        cols  = result.get("columns", list(data[0].keys()))
        header = " | ".join(f"{c.replace('_',' ').title():18s}" for c in cols)
        sep    = "-" * len(header)
        rows   = []
        for row in data[:15]:  # cap display at 15
            rows.append(" | ".join(f"{str(row.get(c,''))[:18]:18s}" for c in cols))

        note = f"\n  *(showing {min(15, row_count)} of {row_count} records)*" if row_count > 15 else ""
        return f"```\n{header}\n{sep}\n" + "\n".join(rows) + f"\n```{note}"

    def _format_rag(self, result: dict) -> str:
        if result.get("type") == "rag_result":
            answer  = result.get("answer", "No answer found.")
            sources = result.get("sources", [])
            src_str = ", ".join(s.replace("_", " ").replace(".txt", "").title() for s in sources)
            footer  = f"\n\n📄 *Source: {src_str}*" if sources else ""
            return f"{answer}{footer}"
        return result.get("message", "No policy information found.")