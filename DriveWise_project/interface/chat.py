"""
interface/chat.py
=================
DriveWise CLI — interactive car brochure assistant.
Maintains conversation history per brand/model session.
"""

import os, sys, json, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Core.rag_engine import DriveWiseRAG
from Core.logger     import QueryLogger, RAGEvaluator


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   🚗  DriveWise — Metadata-Aware Automotive RAG Assistant        ║
║   Ask anything about your selected car from its brochure.        ║
╚══════════════════════════════════════════════════════════════════╝
  Commands:  'brands'   → list available brands
             'models'   → list models for current brand
             'switch'   → switch to a different car
             'history'  → show conversation history
             'sources'  → show last retrieved sources
             'stats'    → show session statistics
             'help'     → show sample questions
             'quit'     → exit
"""

HELP_TEXT = """
  Sample questions you can ask:
  ── Performance ───────────────────────────────────────────────
    What is the engine power and torque?
    How fast can it go from 0 to 100 km/h?
    What transmission options are available?

  ── Efficiency ────────────────────────────────────────────────
    What is the mileage of the petrol variant?
    What is the fuel tank capacity?
    Which variant gives the best fuel efficiency?

  ── Safety ────────────────────────────────────────────────────
    How many airbags does it have?
    What is the Global NCAP rating?
    Does it have ADAS features?

  ── Features ──────────────────────────────────────────────────
    Does it have a sunroof?
    What is the touchscreen size?
    Is wireless Android Auto available?

  ── Pricing ───────────────────────────────────────────────────
    What is the price range?
    What are the available variants and prices?

  ── Dimensions ────────────────────────────────────────────────
    What is the boot space?
    What is the ground clearance?
    What are the overall dimensions?
"""


class ConversationHistory:
    def __init__(self):
        self.turns = []

    def add(self, role: str, content: str, meta: dict = None):
        self.turns.append({
            "role"     : role,
            "content"  : content,
            "meta"     : meta or {},
            "timestamp": datetime.now().isoformat(),
        })

    def display(self, last_n: int = 6):
        recent = self.turns[-last_n:]
        print(f"\n  ── Last {len(recent)} turns ──────────────────────────")
        for t in recent:
            prefix = "  You      " if t["role"] == "user" else "  DriveWise"
            print(f"{prefix}: {t['content'][:100]}")

    def clear(self):
        self.turns = []


class DriveWiseCLI:

    def __init__(self):
        self.rag      = DriveWiseRAG()
        self.logger   = QueryLogger()
        self.evaluator= RAGEvaluator()
        self.history  = ConversationHistory()
        self.brand    = None
        self.model    = None
        self.last_result = None

    def select_car(self):
        """Interactive brand + model selection."""
        brands = self.rag.get_brands()
        print("\n  ── Available Brands ──────────────────────────────")
        for i, b in enumerate(brands, 1):
            print(f"    {i}. {b}")

        while True:
            try:
                choice = input("\n  Select brand number: ").strip()
                idx    = int(choice) - 1
                if 0 <= idx < len(brands):
                    self.brand = brands[idx]
                    break
                print("  Invalid choice. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("  Invalid input.")

        models = self.rag.get_models(self.brand)
        print(f"\n  ── {self.brand} Models ─────────────────────────────")
        for i, m in enumerate(models, 1):
            print(f"    {i}. {m}")

        while True:
            try:
                choice = input("\n  Select model number: ").strip()
                idx    = int(choice) - 1
                if 0 <= idx < len(models):
                    self.model = models[idx]
                    break
                print("  Invalid choice. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("  Invalid input.")

        print(f"\n  ✓ Selected: {self.brand} {self.model}")
        print(f"  You can now ask questions about this car. Type 'help' for examples.\n")

    def _handle_query(self, query: str):
        try:
            result = self.rag.query(query, self.brand, self.model)
            self.last_result = result
            self.logger.log(result)

            # Evaluate response quality
            chunks = []   # we don't expose raw chunks from result dict, that's fine
            metrics = self.evaluator.evaluate(query, result["answer"], [])

            print(f"\n  🚗 DriveWise [{self.brand} {self.model}]:\n")
            print(f"  {result['answer']}\n")
            print(f"  ── Sources ───────────────────────────────────────")
            for s in result["sources"]:
                print(f"    📄 {s['doc_name']} | {s['section']} | Page {s['page_number']}")
            print(f"\n  ⏱  {result['latency_ms']} ms | "
                  f"Faithfulness: {metrics['faithfulness']:.0%} | "
                  f"Relevance: {metrics['context_relevance']:.0%}")

            self.history.add("user",      query)
            self.history.add("assistant", result["answer"],
                             meta={"brand": self.brand, "model": self.model,
                                   "latency_ms": result["latency_ms"]})

        except Exception as e:
            self.logger.log_failed(query, self.brand or "", self.model or "", str(e))
            print(f"\n  ⚠️  Error: {e}\n")

    def run(self):
        print(BANNER)
        self.select_car()

        while True:
            try:
                user_input = input(f"\n  You [{self.brand} {self.model}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Thanks for using DriveWise! Drive safe. 🚗")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "bye"):
                print("\n  Thanks for using DriveWise! Drive safe. 🚗")
                break
            elif cmd == "help":
                print(HELP_TEXT)
            elif cmd == "brands":
                print("\n  Brands:", ", ".join(self.rag.get_brands()))
            elif cmd == "models":
                print(f"\n  {self.brand} models:", ", ".join(self.rag.get_models(self.brand)))
            elif cmd == "switch":
                self.history.clear()
                self.select_car()
            elif cmd == "history":
                self.history.display()
            elif cmd == "sources" and self.last_result:
                print("\n  ── Last Retrieved Sources ────────────────────")
                for s in self.last_result["sources"]:
                    print(f"    📄 Section: {s['section']} | Page {s['page_number']} | Score: {s['score']}")
            elif cmd == "stats":
                stats = self.logger.get_stats()
                avg   = self.evaluator.get_average_metrics()
                print("\n  ── Session Statistics ────────────────────────")
                print(f"    Total queries   : {stats.get('total', 0)}")
                print(f"    Failed queries  : {stats.get('failed', 0)}")
                print(f"    Avg latency     : {stats.get('avg_latency', 0)} ms")
                if avg:
                    print(f"    Avg faithfulness: {avg.get('faithfulness', 0):.0%}")
                    print(f"    Avg relevance   : {avg.get('context_relevance', 0):.0%}")
                    print(f"    Overall score   : {avg.get('overall_score', 0):.0%}")
            else:
                self._handle_query(user_input)


if __name__ == "__main__":
    DriveWiseCLI().run()