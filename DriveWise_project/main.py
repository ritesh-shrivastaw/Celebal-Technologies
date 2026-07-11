"""
main.py — DriveWise Entry Point
================================
Usage:
  python main.py            → setup + interactive CLI chat
  python main.py --setup    → build vector store only
  python main.py --demo     → run 5 demo queries
  python main.py --eval     → run 20-question evaluation suite
  streamlit run app.py      → launch web UI
"""

import os, sys, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DIV = "─" * 65


def phase_setup():
    print(f"\n{'='*65}")
    print("  DRIVEWISE — SETUP")
    print(f"{'='*65}")

    # Step 1: Generate brochures
    print(f"\n  Step 1 — Generating car brochure data...")
    print(DIV)
    from Data.brochure_generator import save_brochures
    brochure_dir = os.path.join(ROOT, "brochures")
    if os.path.exists(os.path.join(brochure_dir, "index.json")):
        print("  ✓ Brochures already exist.")
    else:
        save_brochures(brochure_dir)

    # Step 2: Build vector store
    print(f"\n  Step 2 — Building vector store...")
    print(DIV)
    from Core.rag_engine import build_vector_store, DriveWiseVectorStore
    store_path = os.path.join(ROOT, "vector_store", "drivewise_store.pkl")
    if os.path.exists(store_path):
        print(f"  ✓ Vector store exists at {store_path}")
    else:
        build_vector_store()

    print(f"\n  ✓ Setup complete.\n")


def run_demo():
    from Core.rag_engine import DriveWiseRAG
    rag = DriveWiseRAG()

    DEMO = [
        ("Hyundai",       "Creta",    "What is the mileage of the diesel Creta?"),
        ("Hyundai",       "Creta",    "How many airbags does it have?"),
        ("Maruti Suzuki", "Swift",    "What is the starting price of the Swift?"),
        ("Tata",          "Nexon",    "What is the NCAP safety rating of the Nexon?"),
        ("Mahindra",      "Scorpio-N","Is AWD available on Scorpio-N?"),
    ]

    print(f"\n{'='*65}\n  DRIVEWISE — DEMO QUERIES\n{'='*65}")
    for brand, model, q in DEMO:
        result = rag.query(q, brand, model)
        print(f"\n  [{brand} {model}] {q}")
        print(DIV)
        print(f"  {result['answer'][:300]}")
        print(f"  Sources: {[s['section'] for s in result['sources']]}")
        print(f"  ⏱  {result['latency_ms']:.0f} ms")
    print()


def run_eval():
    from tests.evaluate import run_evaluation
    run_evaluation()


def run_chat():
    from interface.chat import DriveWiseCLI
    DriveWiseCLI().run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriveWise RAG Assistant")
    parser.add_argument("--setup", action="store_true", help="Setup only")
    parser.add_argument("--demo",  action="store_true", help="Demo queries")
    parser.add_argument("--eval",  action="store_true", help="Evaluation suite")
    args = parser.parse_args()

    phase_setup()

    if args.setup:
        pass
    elif args.demo:
        run_demo()
    elif args.eval:
        run_eval()
    else:
        run_chat()