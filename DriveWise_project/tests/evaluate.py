"""
tests/evaluate.py
=================
DriveWise evaluation suite — 20 test questions across brands.
Measures: context relevance, faithfulness, answer correctness, latency.
"""

import os, sys, time, json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Core.rag_engine import DriveWiseRAG
from Core.logger     import RAGEvaluator, QueryLogger

TEST_CASES = [
    # ── Hyundai Creta ─────────────────────────────────────────
    {"brand":"Hyundai","model":"Creta","question":"What is the mileage of the diesel variant?",
     "expected_keywords":["21.8","diesel","km/l"]},
    {"brand":"Hyundai","model":"Creta","question":"How many airbags does the Creta have?",
     "expected_keywords":["6","airbags"]},
    {"brand":"Hyundai","model":"Creta","question":"What is the boot space of the Creta?",
     "expected_keywords":["433"]},
    {"brand":"Hyundai","model":"Creta","question":"What is the price of the top variant?",
     "expected_keywords":["SX","lakh"]},
    {"brand":"Hyundai","model":"Creta","question":"What engine options are available?",
     "expected_keywords":["1.5L","petrol","diesel","turbo"]},

    # ── Hyundai i20 ───────────────────────────────────────────
    {"brand":"Hyundai","model":"i20","question":"What is the fuel efficiency of the i20 diesel?",
     "expected_keywords":["25.05","diesel"]},
    {"brand":"Hyundai","model":"i20","question":"What is the touchscreen size in i20?",
     "expected_keywords":["10.25"]},

    # ── Maruti Swift ──────────────────────────────────────────
    {"brand":"Maruti Suzuki","model":"Swift","question":"What is the mileage of the Swift AMT?",
     "expected_keywords":["25.75","AMT"]},
    {"brand":"Maruti Suzuki","model":"Swift","question":"What is the starting price of Swift?",
     "expected_keywords":["6.49","LXi"]},
    {"brand":"Maruti Suzuki","model":"Swift","question":"Does the Swift have a diesel engine?",
     "expected_keywords":["discontinued","diesel","No"]},

    # ── Maruti Brezza ─────────────────────────────────────────
    {"brand":"Maruti Suzuki","model":"Brezza","question":"What is the ground clearance of Brezza?",
     "expected_keywords":["198"]},
    {"brand":"Maruti Suzuki","model":"Brezza","question":"Does the Brezza have a sunroof?",
     "expected_keywords":["sunroof","ZXi"]},

    # ── Tata Nexon ────────────────────────────────────────────
    {"brand":"Tata","model":"Nexon","question":"What is the Nexon NCAP safety rating?",
     "expected_keywords":["5","star","NCAP"]},
    {"brand":"Tata","model":"Nexon","question":"What is the ground clearance of Nexon?",
     "expected_keywords":["208"]},
    {"brand":"Tata","model":"Nexon","question":"What connected car features does Nexon offer?",
     "expected_keywords":["iRA","features"]},

    # ── Tata Harrier ──────────────────────────────────────────
    {"brand":"Tata","model":"Harrier","question":"What is the diesel engine power of the Harrier?",
     "expected_keywords":["170","PS","2.0L"]},
    {"brand":"Tata","model":"Harrier","question":"What is the boot space in Harrier?",
     "expected_keywords":["425"]},

    # ── Mahindra Scorpio-N ────────────────────────────────────
    {"brand":"Mahindra","model":"Scorpio-N","question":"What is the fuel tank capacity of Scorpio-N?",
     "expected_keywords":["60","litres"]},
    {"brand":"Mahindra","model":"Scorpio-N","question":"Is AWD available on Scorpio-N?",
     "expected_keywords":["AWD","4XPLOR"]},

    # ── Mahindra Thar ─────────────────────────────────────────
    {"brand":"Mahindra","model":"Thar","question":"What is the wading depth of the Thar?",
     "expected_keywords":["650","mm"]},
]


def run_evaluation():
    rag       = DriveWiseRAG()
    evaluator = RAGEvaluator()
    logger    = QueryLogger()
    results   = []

    print("\n" + "="*68)
    print("  DRIVEWISE EVALUATION SUITE — 20 Test Cases")
    print("="*68)

    for i, tc in enumerate(TEST_CASES, 1):
        result = rag.query(tc["question"], tc["brand"], tc["model"])
        logger.log(result)

        # chunks_used is a count; build proxy chunk list for evaluator
        metrics = evaluator.evaluate(
            tc["question"], result["answer"],
            result.get("raw_chunks", []),
            expected_keywords=tc["expected_keywords"]
        )

        kw_hit   = metrics["answer_correctness"] >= 0.33
        faith_ok = metrics["faithfulness"]       >= 0.40
        rel_ok   = metrics["context_relevance"]  >= 0.20

        passed = kw_hit and faith_ok
        status = "✓ PASS" if passed else "✗ FAIL"

        row = {
            **tc,
            **metrics,
            "latency_ms": result["latency_ms"],
            "passed"    : passed,
            "sources"   : [s["section"] for s in result["sources"]],
        }
        results.append(row)

        brand_model = f"{tc['brand']} {tc['model']}"[:22]
        print(f"  {status} | {i:2d} | {brand_model:22s} | "
              f"Corr:{metrics['answer_correctness']:.0%} "
              f"Faith:{metrics['faithfulness']:.0%} "
              f"Rel:{metrics['context_relevance']:.0%} "
              f"| {result['latency_ms']:4.0f}ms")

    # Summary
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    avg_lat = sum(r["latency_ms"] for r in results) / total
    avg_cor = sum(r["answer_correctness"] for r in results) / total
    avg_fai = sum(r["faithfulness"] for r in results) / total
    avg_rel = sum(r["context_relevance"] for r in results) / total

    print(f"\n{'='*68}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*68}")
    print(f"  Pass rate            : {passed}/{total} = {passed/total*100:.1f}%")
    print(f"  Avg latency          : {avg_lat:.1f} ms")
    print(f"  Avg answer correctness: {avg_cor:.0%}")
    print(f"  Avg faithfulness     : {avg_fai:.0%}")
    print(f"  Avg context relevance: {avg_rel:.0%}")
    print(f"{'='*68}\n")

    # Save report
    report_path = os.path.join(ROOT, "logs",
        f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Report saved → {report_path}")
    return results


if __name__ == "__main__":
    run_evaluation()