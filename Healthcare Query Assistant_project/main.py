"""
main.py — RAG-Based Healthcare Query Assistant
Run this file to set up the full pipeline and launch the assistant.

Usage:
  python main.py           → full setup + interactive chat
  python main.py --setup   → only run Phase 1 setup (data + KB)
  python main.py --eval    → only run evaluation suite
  python main.py --demo    → run 5 demo queries non-interactively
"""

import os
import sys
import argparse

# ── Ensure project root is in path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DIVIDER = "─" * 65


def phase1_setup():
    """Run Phase 1: generate data, build DB, build vector knowledge base."""
    print(f"\n{'='*65}")
    print("  PHASE 1 — DATA PREPARATION")
    print(f"{'='*65}")

    # Part 1: Generate dataset + SQLite DB
    print(f"\n  Part 1 — Synthetic Dataset & Database")
    print(DIVIDER)
    from data.generate_dataset import generate_patients, build_database, validate_database
    db_path  = os.path.join(ROOT, "database/healthcare.db")
    csv_path = os.path.join(ROOT, "data/patients.csv")

    if os.path.exists(db_path):
        print(f"  ✓ Database already exists at {db_path}")
        print(f"    Delete it to regenerate.")
    else:
        df = generate_patients()
        df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV saved → {csv_path}")
        build_database(df, db_path)
        validate_database(db_path)

    # Part 2: Build vector knowledge base
    print(f"\n  Part 2 — Hospital Policy Knowledge Base")
    print(DIVIDER)
    from knowledge_base.policy_generator import build_knowledge_base
    index_path = os.path.join(ROOT, "knowledge_base/faiss_index")

    if os.path.exists(os.path.join(index_path, "store.pkl")):
        print(f"  ✓ Knowledge base already exists at {index_path}")
        print(f"    Delete it to regenerate.")
    else:
        build_knowledge_base()

    print(f"\n  ✓ Phase 1 complete.\n")


def run_demo():
    """Run 5 pre-set demo queries and print results."""
    from agents.agents import (OrchestratorAgent, NLPtoSQLAgent,
                                RAGAgent, ResponseFormatter)

    orchestrator = OrchestratorAgent()
    sql_agent    = NLPtoSQLAgent()
    rag_agent    = RAGAgent()
    formatter    = ResponseFormatter()

    demo_queries = [
        "How many diabetic patients are there?",
        "Which patients have abnormal test results?",
        "Show the average billing amount by insurance provider.",
        "What is the hospital discharge policy?",
        "Is prior insurance authorisation required for surgery?",
    ]

    print(f"\n{'='*65}")
    print("  DEMO — 5 Sample Queries")
    print(f"{'='*65}")

    for i, query in enumerate(demo_queries, 1):
        print(f"\n  [{i}] {query}")
        print(DIVIDER)
        intent   = orchestrator.classify_intent(query)
        response = orchestrator.route(query, sql_agent, rag_agent, formatter)
        print(f"  Intent   : {intent.upper()}")
        print(f"  Response :\n{response}")

    print(f"\n{'='*65}\n")


def run_evaluation():
    """Run the full evaluation suite."""
    from tests.evaluate import Evaluator
    Evaluator().run_all()


def main():
    parser = argparse.ArgumentParser(description="RAG Healthcare Query Assistant")
    parser.add_argument("--setup", action="store_true", help="Run Phase 1 setup only")
    parser.add_argument("--eval",  action="store_true", help="Run evaluation suite only")
    parser.add_argument("--demo",  action="store_true", help="Run 5 demo queries")
    args = parser.parse_args()

    if args.setup:
        phase1_setup()

    elif args.eval:
        phase1_setup()
        run_evaluation()

    elif args.demo:
        phase1_setup()
        run_demo()

    else:
        # Full flow: setup + interactive chat
        phase1_setup()
        from interface.chat import HealthcareAssistant
        assistant = HealthcareAssistant()
        assistant.run()


if __name__ == "__main__":
    main()