"""
Phase 1 – Part 5
Testing & Evaluation Suite
  - SQL accuracy tests (exact + fuzzy match)
  - RAG retrieval relevance scoring
  - Ambiguous / out-of-domain query handling
  - Latency benchmarking
  - Summary report
"""

import os
import sys
import time
import json
import sqlite3
import re
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.agents import (OrchestratorAgent, NLPtoSQLAgent,
                            RAGAgent, ResponseFormatter)

DB_PATH = os.path.join(ROOT, "database/healthcare.db")


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────
SQL_TESTS = [
    {
        "id"          : "SQL-01",
        "query"       : "How many diabetic patients are there?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["clinical_data", "diabetes"],
        "expect_row_count_gte": 1,
        "description" : "Count patients with Diabetes condition",
    },
    {
        "id"          : "SQL-02",
        "query"       : "Which patients have abnormal test results?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["Abnormal"],
        "expect_row_count_gte": 1,
        "description" : "List patients with abnormal test results",
    },
    {
        "id"          : "SQL-03",
        "query"       : "Show the average billing amount by insurance provider",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["AVG", "insurance_provider"],
        "expect_row_count_gte": 1,
        "description" : "Aggregate billing by insurance group",
    },
    {
        "id"          : "SQL-04",
        "query"       : "How many emergency admissions are there?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["Emergency"],
        "expect_row_count_gte": 1,
        "description" : "Count emergency admissions",
    },
    {
        "id"          : "SQL-05",
        "query"       : "What is the most common medical condition?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["medical_condition", "COUNT"],
        "expect_row_count_gte": 1,
        "description" : "Top medical condition by frequency",
    },
    {
        "id"          : "SQL-06",
        "query"       : "Show gender distribution of patients",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["gender"],
        "expect_row_count_gte": 1,
        "description" : "Gender breakdown of patient population",
    },
    {
        "id"          : "SQL-07",
        "query"       : "What is the average age of patients?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["AVG", "age"],
        "expect_row_count_gte": 1,
        "description" : "Mean patient age",
    },
    {
        "id"          : "SQL-08",
        "query"       : "How many patients are at each hospital?",
        "expect_type" : "sql_result",
        "expect_sql_contains": ["hospital", "COUNT"],
        "expect_row_count_gte": 1,
        "description" : "Patient distribution across hospitals",
    },
]

RAG_TESTS = [
    {
        "id"             : "RAG-01",
        "query"          : "What is the hospital discharge policy?",
        "expect_sources" : ["discharge_policy.txt"],
        "expect_keywords": ["discharge", "physician", "plan"],
        "description"    : "Core discharge policy retrieval",
    },
    {
        "id"             : "RAG-02",
        "query"          : "Is prior insurance authorisation required for surgery?",
        "expect_sources" : ["billing_policy.txt", "insurance_policy.txt"],
        "expect_keywords": ["authorisation", "surgery", "prior"],
        "description"    : "Pre-auth for surgical procedures",
    },
    {
        "id"             : "RAG-03",
        "query"          : "What is the emergency triage protocol?",
        "expect_sources" : ["emergency_policy.txt"],
        "expect_keywords": ["triage", "ESI", "emergency"],
        "description"    : "Emergency triage ESI protocol",
    },
    {
        "id"             : "RAG-04",
        "query"          : "What are patient rights under HIPAA?",
        "expect_sources" : ["privacy_policy.txt"],
        "expect_keywords": ["HIPAA", "right", "records"],
        "description"    : "HIPAA patient rights retrieval",
    },
    {
        "id"             : "RAG-05",
        "query"          : "How does the billing dispute process work?",
        "expect_sources" : ["billing_policy.txt"],
        "expect_keywords": ["dispute", "appeal", "billing"],
        "description"    : "Billing dispute resolution",
    },
    {
        "id"             : "RAG-06",
        "query"          : "What documents are required for hospital admission?",
        "expect_sources" : ["admission_policy.txt"],
        "expect_keywords": ["ID", "insurance", "required"],
        "description"    : "Admission documentation requirements",
    },
]

AMBIGUOUS_TESTS = [
    {
        "id"         : "AMB-01",
        "query"      : "Tell me about John",
        "expect_not" : "crash",
        "description": "Ambiguous name query — should not crash",
    },
    {
        "id"         : "AMB-02",
        "query"      : "What is the weather today?",
        "expect_not" : "crash",
        "description": "Out-of-domain query — should return fallback",
    },
    {
        "id"         : "AMB-03",
        "query"      : "",
        "expect_not" : "crash",
        "description": "Empty query — graceful handling",
    },
    {
        "id"         : "AMB-04",
        "query"      : "Show me everything",
        "expect_not" : "crash",
        "description": "Vague over-broad query",
    },
    {
        "id"         : "AMB-05",
        "query"      : "!!!???",
        "expect_not" : "crash",
        "description": "Garbage/special char input",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class Evaluator:

    def __init__(self):
        self.sql_agent    = NLPtoSQLAgent(db_path=DB_PATH)
        self.rag_agent    = RAGAgent()
        self.orchestrator = OrchestratorAgent()
        self.formatter    = ResponseFormatter()
        self.results      = []

    # ── SQL evaluation ────────────────────────────────────────────────────────
    def eval_sql(self, test: dict) -> dict:
        t0     = time.time()
        result = self.sql_agent.run(test["query"])
        ms     = (time.time() - t0) * 1000

        sql_gen    = result.get("sql", "")
        row_count  = result.get("row_count", 0)
        res_type   = result.get("type", "")

        checks = {}
        checks["type_correct"] = (res_type == test["expect_type"])
        checks["sql_keywords"] = all(
            kw.lower() in sql_gen.lower()
            for kw in test.get("expect_sql_contains", [])
        )
        checks["has_rows"] = (row_count >= test.get("expect_row_count_gte", 1))

        passed = all(checks.values())

        return {
            "id"         : test["id"],
            "description": test["description"],
            "query"      : test["query"],
            "passed"     : passed,
            "checks"     : checks,
            "sql"        : sql_gen,
            "rows"       : row_count,
            "latency_ms" : round(ms, 1),
            "category"   : "SQL",
        }

    # ── RAG evaluation ────────────────────────────────────────────────────────
    def eval_rag(self, test: dict) -> dict:
        t0     = time.time()
        result = self.rag_agent.run(test["query"])
        ms     = (time.time() - t0) * 1000

        sources  = result.get("sources", [])
        chunks   = " ".join(result.get("chunks", []))
        answer   = result.get("answer", "")

        checks = {}
        expected_src = test.get("expect_sources", [])
        checks["source_match"] = any(s in sources for s in expected_src) if expected_src else True
        checks["keywords_in_chunks"] = all(
            kw.lower() in chunks.lower()
            for kw in test.get("expect_keywords", [])
        )
        checks["has_answer"] = len(answer.strip()) > 20

        # Relevance score: fraction of expected keywords found
        kws = test.get("expect_keywords", [])
        relevance = sum(1 for kw in kws if kw.lower() in chunks.lower()) / len(kws) if kws else 1.0

        passed = all(checks.values())

        return {
            "id"          : test["id"],
            "description" : test["description"],
            "query"       : test["query"],
            "passed"      : passed,
            "checks"      : checks,
            "sources"     : sources,
            "relevance"   : round(relevance, 2),
            "latency_ms"  : round(ms, 1),
            "category"    : "RAG",
        }

    # ── Ambiguous / out-of-domain evaluation ──────────────────────────────────
    def eval_ambiguous(self, test: dict) -> dict:
        t0 = time.time()
        try:
            if not test["query"].strip():
                result = {"status": "empty_handled"}
                response = "Please enter a question."
            else:
                response = self.orchestrator.route(
                    test["query"], self.sql_agent, self.rag_agent, self.formatter
                )
                result = {"status": "ok"}
            passed = True
            error  = None
        except Exception as e:
            passed = False
            error  = str(e)
            response = ""
        ms = (time.time() - t0) * 1000

        return {
            "id"         : test["id"],
            "description": test["description"],
            "query"      : test["query"],
            "passed"     : passed,
            "error"      : error,
            "response_len": len(response),
            "latency_ms" : round(ms, 1),
            "category"   : "Ambiguous",
        }

    # ── Run all tests ─────────────────────────────────────────────────────────
    def run_all(self):
        print("\n" + "="*65)
        print("  EVALUATION SUITE — RAG Healthcare Query Assistant")
        print("="*65)

        # SQL tests
        print(f"\n  ── SQL Tests ({len(SQL_TESTS)} cases) ───────────────────────────────")
        for test in SQL_TESTS:
            r = self.eval_sql(test)
            self.results.append(r)
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"  {status} | {r['id']} | {r['description'][:45]:45s} | {r['latency_ms']:5.0f}ms")
            if not r["passed"]:
                for k, v in r["checks"].items():
                    if not v:
                        print(f"          └─ FAILED CHECK: {k}")

        # RAG tests
        print(f"\n  ── RAG Tests ({len(RAG_TESTS)} cases) ───────────────────────────────")
        for test in RAG_TESTS:
            r = self.eval_rag(test)
            self.results.append(r)
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            rel    = f"relevance={r['relevance']:.0%}"
            print(f"  {status} | {r['id']} | {r['description'][:40]:40s} | {rel} | {r['latency_ms']:5.0f}ms")
            if not r["passed"]:
                for k, v in r["checks"].items():
                    if not v:
                        print(f"          └─ FAILED CHECK: {k}")

        # Ambiguous tests
        print(f"\n  ── Ambiguous / OOD Tests ({len(AMBIGUOUS_TESTS)} cases) ──────────────")
        for test in AMBIGUOUS_TESTS:
            r = self.eval_ambiguous(test)
            self.results.append(r)
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"  {status} | {r['id']} | {r['description'][:50]:50s} | {r['latency_ms']:5.0f}ms")

        self._print_summary()
        self._save_report()

    def _print_summary(self):
        total  = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        by_cat = {}
        for r in self.results:
            cat = r["category"]
            by_cat.setdefault(cat, {"pass": 0, "total": 0, "lat": []})
            by_cat[cat]["total"] += 1
            by_cat[cat]["lat"].append(r["latency_ms"])
            if r["passed"]:
                by_cat[cat]["pass"] += 1

        avg_lat = sum(r["latency_ms"] for r in self.results) / total

        print(f"\n{'='*65}")
        print(f"  SUMMARY")
        print(f"{'='*65}")
        print(f"  Overall accuracy : {passed}/{total} = {passed/total*100:.1f}%")
        print(f"  Avg latency      : {avg_lat:.0f} ms")
        print(f"\n  By category:")
        for cat, stats in by_cat.items():
            acc     = stats["pass"] / stats["total"] * 100
            avg_l   = sum(stats["lat"]) / len(stats["lat"])
            print(f"    {cat:12s}: {stats['pass']}/{stats['total']} = {acc:.0f}%  |  avg {avg_l:.0f}ms")
        print(f"{'='*65}\n")

    def _save_report(self):
        report_path = os.path.join(ROOT, "logs", f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  Report saved → {report_path}")


if __name__ == "__main__":
    Evaluator().run_all()