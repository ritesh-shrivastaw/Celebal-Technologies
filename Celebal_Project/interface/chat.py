"""
Phase 1 – Part 4
Conversational Interface
  - Maintains full conversation history
  - Tracks agent routing decisions
  - Error handling + fallback responses
  - Session logging
"""

import os
import json
import time
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "conversation.log"),
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)


class ConversationMemory:
    """Stores the full conversation history for the session."""

    def __init__(self, max_turns: int = 20):
        self.history   : list[dict] = []
        self.max_turns : int        = max_turns

    def add(self, role: str, content: str, meta: dict = None):
        self.history.append({
            "role"      : role,
            "content"   : content,
            "timestamp" : datetime.now().isoformat(),
            "meta"      : meta or {},
        })
        # Keep only last N turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context(self, last_n: int = 4) -> str:
        """Return the last N exchanges as a readable string."""
        recent = self.history[-(last_n * 2):]
        lines  = []
        for turn in recent:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content'][:200]}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return self.history

    def clear(self):
        self.history = []
        print("  [Memory cleared]")


class RoutingLogger:
    """Records every agent routing decision with latency."""

    def __init__(self):
        self.decisions: list[dict] = []

    def log(self, query: str, intent: str, agent: str, latency_ms: float):
        entry = {
            "query"      : query,
            "intent"     : intent,
            "agent"      : agent,
            "latency_ms" : round(latency_ms, 1),
            "timestamp"  : datetime.now().isoformat(),
        }
        self.decisions.append(entry)
        logging.info(f"ROUTE | intent={intent} | agent={agent} | latency={latency_ms:.0f}ms | query='{query[:60]}'")

    def summary(self) -> str:
        if not self.decisions:
            return "No routing decisions recorded."
        total   = len(self.decisions)
        intents = {}
        avg_lat = sum(d["latency_ms"] for d in self.decisions) / total
        for d in self.decisions:
            intents[d["intent"]] = intents.get(d["intent"], 0) + 1
        lines = [
            f"  Total queries   : {total}",
            f"  Avg latency     : {avg_lat:.0f} ms",
            f"  Intent breakdown:",
        ]
        for intent, count in intents.items():
            lines.append(f"    {intent:12s}: {count} ({count/total*100:.0f}%)")
        return "\n".join(lines)


class HealthcareAssistant:
    """
    Main conversational interface.
    Wires together: OrchestratorAgent → NLPtoSQLAgent / RAGAgent → ResponseFormatter
    """

    BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        RAG-Based Healthcare Query Assistant                  ║
║        Multi-Agent System  |  Hospital Knowledge Base        ║
╚══════════════════════════════════════════════════════════════╝
  Type your question in natural language.
  Commands: 'history'  → show conversation history
            'routing'  → show agent routing stats
            'clear'    → clear conversation memory
            'help'     → show sample queries
            'quit'     → exit
"""

    HELP_TEXT = """
  Sample queries you can try:
  ── Patient Data (→ SQL Agent) ───────────────────────────────
    • How many diabetic patients were admitted last month?
    • Which patients have abnormal test results?
    • Show the average billing amount by insurance provider.
    • How many emergency admissions are there?
    • What is the most common medical condition?
    • Show gender distribution of patients.
    • What is the average age of patients?
    • How many patients are at each hospital?

  ── Hospital Policy (→ RAG Agent) ───────────────────────────
    • What is the hospital discharge policy?
    • Is prior insurance authorisation required for surgery?
    • What is the emergency triage protocol?
    • What are the patient rights under HIPAA?
    • How does the billing dispute process work?
    • What documents are required for admission?
    • What is the hospital's no-refusal policy?
"""

    def __init__(self):
        from agents.agents import (OrchestratorAgent, NLPtoSQLAgent,
                                   RAGAgent, ResponseFormatter)
        self.orchestrator = OrchestratorAgent()
        self.sql_agent    = NLPtoSQLAgent()
        self.rag_agent    = RAGAgent()
        self.formatter    = ResponseFormatter()
        self.memory       = ConversationMemory()
        self.router_log   = RoutingLogger()
        self.session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _handle(self, query: str) -> str:
        """Route query and return formatted response with timing."""
        start  = time.time()
        intent = self.orchestrator.classify_intent(query)

        try:
            response = self.orchestrator.route(
                query, self.sql_agent, self.rag_agent, self.formatter
            )
        except Exception as e:
            logging.error(f"ERROR | session={self.session_id} | {e}")
            response = (
                "⚠️  I encountered an error processing your request. "
                "Please try rephrasing or ask a different question."
            )

        latency = (time.time() - start) * 1000
        self.router_log.log(query, intent, intent, latency)
        self.memory.add("user",      query)
        self.memory.add("assistant", response, meta={"intent": intent, "latency_ms": latency})
        return response, latency, intent

    def run(self):
        """Start the interactive CLI loop."""
        print(self.BANNER)

        while True:
            try:
                user_input = input("\n  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye! Stay healthy. 👋")
                self._save_session()
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "bye"):
                print("\n  Goodbye! Stay healthy. 👋")
                self._save_session()
                break

            elif cmd == "history":
                print("\n  ── Conversation History ──────────────────────")
                for turn in self.memory.history[-10:]:
                    prefix = "  You      " if turn["role"] == "user" else "  Assistant"
                    print(f"{prefix}: {turn['content'][:120]}")
                continue

            elif cmd == "routing":
                print("\n  ── Routing Statistics ────────────────────────")
                print(self.router_log.summary())
                continue

            elif cmd == "clear":
                self.memory.clear()
                continue

            elif cmd == "help":
                print(self.HELP_TEXT)
                continue

            # Normal query
            print("\n  Assistant: ", end="", flush=True)
            response, latency, intent = self._handle(user_input)
            print(response)
            print(f"\n  ── [{intent.upper()} | {latency:.0f}ms] ──")

    def _save_session(self):
        """Persist conversation history to logs."""
        path = os.path.join(LOG_DIR, f"session_{self.session_id}.json")
        with open(path, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "history"   : self.memory.to_dict(),
                "routing"   : self.router_log.decisions,
            }, f, indent=2)
        print(f"\n  Session saved → {path}")


if __name__ == "__main__":
    assistant = HealthcareAssistant()
    assistant.run()