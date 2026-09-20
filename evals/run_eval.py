
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gate import decide
from src.retrieval import Retriever

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def run() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()

    recall_hits, recall_total = 0, 0
    route_hits, route_total = 0, 0
    failures = []

    for case in cases:
        results = retriever.search(case["query"], k=4)
        got_ids = {s.record.id for s in results}
        expected_any = set(case.get("expected_sources_any", []))

        if expected_any:
            recall_total += 1
            hit = bool(got_ids & expected_any)
            recall_hits += hit
            if not hit:
                failures.append(
                    f"[recall] {case['id']}: expected one of {expected_any}, got {got_ids}"
                )

        gate = decide(case["query"], results)
        route_total += 1
        route_ok = gate.route.value == case["expected_route"]
        route_hits += route_ok
        if not route_ok:
            failures.append(
                f"[route] {case['id']}: expected {case['expected_route']}, got {gate.route.value} "
                f"(top_score={gate.top_score:.3f})"
            )

    recall_score = recall_hits / recall_total if recall_total else 1.0
    route_score = route_hits / route_total if route_total else 1.0

    print(f"Retrieval recall@4:        {recall_hits}/{recall_total}  ({recall_score:.0%})")
    print(f"Escalation routing accuracy: {route_hits}/{route_total}  ({route_score:.0%})")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll cases passed.")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())
