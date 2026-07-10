"""
agents/langchain_agents.py
Production-grade agents using LangChain + OpenAI.
Drop-in replacements for the rule-based agents in agents.py.

Requirements:
  pip install langchain langchain-community langchain-openai
  faiss-cpu sentence-transformers openai
  Set OPENAI_API_KEY in your .env file.
"""

import os
import sys
import sqlite3
import logging
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Guard: only import LangChain if available ────────────────────────────────
def _check_deps():
    missing = []
    try: import langchain
    except ImportError: missing.append("langchain")
    try: import langchain_openai
    except ImportError: missing.append("langchain-openai")
    try: import langchain_community
    except ImportError: missing.append("langchain-community")
    try: import sentence_transformers
    except ImportError: missing.append("sentence-transformers")
    try: import faiss
    except ImportError: missing.append("faiss-cpu")
    if missing:
        raise ImportError(
            f"Missing packages for production agents: {', '.join(missing)}\n"
            f"Run: pip install {' '.join(missing)}"
        )

_check_deps()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import create_sql_query_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document


DB_PATH      = os.path.join(ROOT, "database", "healthcare.db")
POLICIES_DIR = os.path.join(ROOT, "knowledge_base", "policies")
FAISS_PATH   = os.path.join(ROOT, "knowledge_base", "faiss_openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K      = int(os.getenv("RAG_TOP_K", 4))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 80))


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION NLP-TO-SQL AGENT (LangChain SQLDatabaseChain)
# ═══════════════════════════════════════════════════════════════════════════════
class LangChainSQLAgent:
    """
    Uses LangChain's create_sql_query_chain with GPT-4o-mini.
    Supports any natural language question against the SQLite schema.
    """

    SYSTEM_PROMPT = """You are an expert SQL assistant for a hospital database.
Generate a valid SQLite query to answer the user's question.

Database schema:
  patients(patient_id, name, age, gender, blood_type)
  clinical_data(record_id, patient_id, medical_condition, medication, test_results)
  admissions(admission_id, patient_id, hospital, doctor, room_number,
             admission_type, admission_date, discharge_date)
  billing(billing_id, patient_id, insurance_provider, billing_amount)

Rules:
- Always JOIN tables via patient_id when cross-table data is needed.
- Use LOWER() and LIKE for case-insensitive text matching.
- Limit results to 20 rows unless a count/aggregate is requested.
- Return ONLY the SQL query, no explanation, no markdown backticks.
"""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{DB_PATH}",
            include_tables=["patients", "clinical_data", "admissions", "billing"],
        )
        self.chain = create_sql_query_chain(self.llm, self.db)

    def run(self, query: str) -> dict:
        import time
        t0 = time.time()
        try:
            # Step 1: Generate SQL
            sql = self.chain.invoke({"question": query})
            sql = sql.strip().strip("```sql").strip("```").strip()

            # Step 2: Execute
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows    = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()

            data = [dict(zip(columns, row)) for row in rows]
            ms   = (time.time() - t0) * 1000
            logging.info(f"LangChain SQL | {ms:.0f}ms | rows={len(data)} | sql={sql[:80]}")
            return {"type": "sql_result", "sql": sql, "data": data,
                    "row_count": len(data), "columns": columns}

        except Exception as e:
            logging.error(f"LangChain SQL ERROR | {e}")
            return {"type": "sql_error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION RAG AGENT (FAISS + OpenAI Embeddings + GPT-4o-mini)
# ═══════════════════════════════════════════════════════════════════════════════
class LangChainRAGAgent:
    """
    FAISS vector store with OpenAI embeddings + GPT-4o-mini for answer generation.
    Builds the index on first run; loads from disk on subsequent runs.
    """

    ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a hospital policy assistant. Answer the question using ONLY the provided
policy document excerpts. Be concise and specific. If the answer is not covered
in the excerpts, say "This is not covered in the available policy documents."

Policy excerpts:
{context}

Question: {question}

Answer:""")

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        self.llm        = ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
        self._vectorstore = None

    def _load_or_build_index(self):
        if self._vectorstore is not None:
            return
        if os.path.exists(FAISS_PATH):
            print("  Loading FAISS index from disk…")
            self._vectorstore = FAISS.load_local(
                FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print("  Building FAISS index with OpenAI embeddings…")
            self._build_index()

    def _build_index(self):
        # Load all policy .txt files
        docs = []
        for fname in os.listdir(POLICIES_DIR):
            if not fname.endswith(".txt"):
                continue
            fpath   = os.path.join(POLICIES_DIR, fname)
            with open(fpath) as f:
                content = f.read()
            docs.append(Document(
                page_content=content,
                metadata={"source": fname, "doc_type": fname.replace("_policy.txt","").title()}
            ))

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(docs)
        print(f"  Chunked {len(docs)} documents → {len(chunks)} chunks")

        # Build FAISS index
        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self._vectorstore.save_local(FAISS_PATH)
        print(f"  FAISS index saved → {FAISS_PATH}")

    def run(self, query: str, top_k: int = RAG_TOP_K) -> dict:
        import time
        t0 = time.time()
        self._load_or_build_index()

        retriever = self._vectorstore.as_retriever(
            search_type="mmr",            # Maximal Marginal Relevance for diversity
            search_kwargs={"k": top_k, "fetch_k": top_k * 3},
        )

        # Build RAG chain
        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[{d.metadata.get('source','Unknown')}]\n{d.page_content}"
                for d in docs
            )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )

        try:
            answer  = rag_chain.invoke(query)
            docs    = retriever.invoke(query)
            sources = list({d.metadata["source"] for d in docs})
            chunks  = [d.page_content for d in docs]
            ms      = (time.time() - t0) * 1000
            logging.info(f"LangChain RAG | {ms:.0f}ms | sources={sources}")
            return {"type": "rag_result", "answer": answer,
                    "sources": sources, "chunks": chunks}
        except Exception as e:
            logging.error(f"LangChain RAG ERROR | {e}")
            return {"type": "rag_result", "answer": f"Error: {e}", "sources": [], "chunks": []}


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION ORCHESTRATOR (LLM-based intent classification)
# ═══════════════════════════════════════════════════════════════════════════════
class LangChainOrchestratorAgent:
    """
    Uses GPT-4o-mini to classify intent — more accurate than keyword matching,
    handles ambiguous and complex queries gracefully.
    """

    CLASSIFY_PROMPT = PromptTemplate.from_template("""
Classify this hospital assistant query into exactly one category:

- "database"  → asks about specific patient records, counts, statistics,
                 billing amounts, admissions, test results (needs SQL)
- "policy"    → asks about hospital rules, procedures, discharge process,
                 insurance requirements, HIPAA, triage protocols (needs RAG)
- "hybrid"    → needs both patient data AND policy context
- "unknown"   → completely unrelated to healthcare or too vague

Query: {query}

Respond with ONLY the category word (database / policy / hybrid / unknown):
""")

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
        self.chain = self.CLASSIFY_PROMPT | self.llm | StrOutputParser()

    def classify_intent(self, query: str) -> str:
        if not query.strip():
            return "unknown"
        try:
            result = self.chain.invoke({"query": query}).strip().lower()
            return result if result in ("database","policy","hybrid","unknown") else "unknown"
        except Exception as e:
            logging.error(f"Orchestrator classify error: {e}")
            # Fallback to keyword-based
            from agents.agents import OrchestratorAgent
            return OrchestratorAgent().classify_intent(query)

    def route(self, query: str, sql_agent, rag_agent, formatter) -> str:
        intent = self.classify_intent(query)
        if intent == "database":
            result = sql_agent.run(query)
        elif intent == "policy":
            result = rag_agent.run(query)
        elif intent == "hybrid":
            db_res  = sql_agent.run(query)
            rag_res = rag_agent.run(query)
            result  = {"type": "hybrid", "database": db_res, "policy": rag_res}
        else:
            result = {"type": "fallback",
                      "message": "I can help with patient records or hospital policies. "
                                 "Please rephrase your question."}
        return formatter.format(result, intent, query)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY — swap rule-based ↔ LangChain agents via environment variable
# ═══════════════════════════════════════════════════════════════════════════════
def get_agents(use_llm: bool = None):
    """
    Returns the right set of agents based on environment.
    use_llm=True  → LangChain + OpenAI (production)
    use_llm=False → rule-based (no API key needed, default)
    use_llm=None  → auto-detect from OPENAI_API_KEY env var
    """
    from agents.agents import ResponseFormatter

    if use_llm is None:
        use_llm = bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))

    if use_llm:
        print("  Using LangChain production agents (LLM-backed)")
        return {
            "orchestrator" : LangChainOrchestratorAgent(),
            "sql"          : LangChainSQLAgent(),
            "rag"          : LangChainRAGAgent(),
            "formatter"    : ResponseFormatter(),
            "mode"         : "production",
        }
    else:
        from agents.agents import OrchestratorAgent, NLPtoSQLAgent, RAGAgent
        print("  Using rule-based agents (no API key)")
        return {
            "orchestrator" : OrchestratorAgent(),
            "sql"          : NLPtoSQLAgent(),
            "rag"          : RAGAgent(),
            "formatter"    : ResponseFormatter(),
            "mode"         : "rule-based",
        }


if __name__ == "__main__":
    # Quick smoke test (requires OPENAI_API_KEY)
    agents = get_agents(use_llm=True)
    queries = [
        "How many diabetic patients are there?",
        "What is the hospital discharge policy?",
    ]
    formatter = agents["formatter"]
    for q in queries:
        print(f"\nQ: {q}")
        intent = agents["orchestrator"].classify_intent(q)
        resp   = agents["orchestrator"].route(q, agents["sql"], agents["rag"], formatter)
        print(f"Intent: {intent}")
        print(f"Response: {resp[:300]}")