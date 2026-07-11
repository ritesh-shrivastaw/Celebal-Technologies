"""
core/logger.py
==============
DriveWise logging + evaluation system.
Tracks: queries, latency, failed queries, retrieval results,
        answer correctness, faithfulness, context relevance.
"""

import os, json, re, time
from datetime import datetime

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

QUERY_LOG   = os.path.join(LOG_DIR, "query_log.jsonl")
FAILED_LOG  = os.path.join(LOG_DIR, "failed_queries.jsonl")
METRICS_LOG = os.path.join(LOG_DIR, "eval_metrics.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY LOGGER
# ═══════════════════════════════════════════════════════════════════════════════
class QueryLogger:
    """Records every query with full metadata."""

    def log(self, result: dict, status: str = "success"):
        entry = {
            "timestamp"   : datetime.now().isoformat(),
            "status"      : status,
            "brand"       : result.get("brand"),
            "model"       : result.get("model"),
            "question"    : result.get("question"),
            "latency_ms"  : result.get("latency_ms"),
            "chunks_used" : result.get("chunks_used"),
            "sources"     : [s["section"] for s in result.get("sources", [])],
        }
        with open(QUERY_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_failed(self, question: str, brand: str, model: str, error: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "brand"    : brand,
            "model"    : model,
            "question" : question,
            "error"    : error,
        }
        with open(FAILED_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_stats(self) -> dict:
        """Return session statistics from log file."""
        if not os.path.exists(QUERY_LOG):
            return {}
        entries = []
        with open(QUERY_LOG) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

        if not entries:
            return {"total": 0}

        latencies = [e["latency_ms"] for e in entries if e.get("latency_ms")]
        brands    = {}
        for e in entries:
            b = e.get("brand", "Unknown")
            brands[b] = brands.get(b, 0) + 1

        failed = 0
        if os.path.exists(FAILED_LOG):
            with open(FAILED_LOG) as f:
                failed = sum(1 for _ in f)

        return {
            "total"       : len(entries),
            "failed"      : failed,
            "avg_latency" : round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "min_latency" : round(min(latencies), 1) if latencies else 0,
            "max_latency" : round(max(latencies), 1) if latencies else 0,
            "by_brand"    : brands,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR — answer correctness, faithfulness, context relevance
# ═══════════════════════════════════════════════════════════════════════════════
class RAGEvaluator:
    """
    Lightweight evaluation without an external LLM.
    Measures three RAG quality dimensions:

    1. Context Relevance  — are retrieved chunks relevant to the query?
    2. Faithfulness       — does the answer stay within the retrieved context?
    3. Answer Correctness — does the answer contain expected key terms?

    Production: Replace with RAGAS framework
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    """

    def evaluate(self, question: str, answer: str,
                 context_chunks: list[dict], expected_keywords: list[str] = None) -> dict:

        context_text = " ".join(c["text"] for c in context_chunks).lower()
        answer_lower = answer.lower()
        q_words      = set(re.findall(r"[a-z0-9]+", question.lower()))

        # ── 1. Context Relevance ─────────────────────────────────────────────
        # How many query terms appear in retrieved context?
        ctx_hits = sum(1 for w in q_words if len(w) > 3 and w in context_text)
        context_relevance = round(min(ctx_hits / max(len(q_words), 1), 1.0), 3)

        # ── 2. Faithfulness ──────────────────────────────────────────────────
        # What fraction of answer sentences are grounded in context?
        answer_sentences = [s.strip() for s in re.split(r"[.•\n]", answer) if len(s.strip()) > 15]
        if not answer_sentences:
            faithfulness = 1.0
        else:
            grounded = 0
            for sent in answer_sentences:
                sent_words = set(re.findall(r"[a-z0-9]+", sent.lower()))
                overlap    = len(sent_words & set(re.findall(r"[a-z0-9]+", context_text)))
                if overlap >= 2:
                    grounded += 1
            faithfulness = round(grounded / len(answer_sentences), 3)

        # ── 3. Answer Correctness ────────────────────────────────────────────
        # If expected_keywords given, check how many appear in answer
        if expected_keywords:
            # Flexible match: keyword OR any word within it appears in answer
            def kw_match(kw, text):
                kw = kw.lower()
                if kw in text: return True
                # Also match individual words of multi-word keywords
                for part in kw.split():
                    if len(part) > 2 and part in text: return True
                return False
            hits = sum(1 for kw in expected_keywords if kw_match(kw, answer_lower))
            answer_correctness = round(hits / len(expected_keywords), 3)
        else:
            # Proxy: answer contains numeric values (specs) → more informative
            nums = re.findall(r"\d+\.?\d*", answer)
            answer_correctness = min(len(nums) / 3, 1.0)
            answer_correctness = round(answer_correctness, 3)

        score = round((context_relevance + faithfulness + answer_correctness) / 3, 3)

        metrics = {
            "context_relevance" : context_relevance,
            "faithfulness"      : faithfulness,
            "answer_correctness": answer_correctness,
            "overall_score"     : score,
        }

        # Log metrics
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question" : question,
            **metrics,
        }
        with open(METRICS_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return metrics

    def get_average_metrics(self) -> dict:
        if not os.path.exists(METRICS_LOG):
            return {}
        entries = []
        with open(METRICS_LOG) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        if not entries:
            return {}
        keys = ["context_relevance", "faithfulness", "answer_correctness", "overall_score"]
        return {
            k: round(sum(e.get(k, 0) for e in entries) / len(entries), 3)
            for k in keys
        } | {"total_evaluated": len(entries)}