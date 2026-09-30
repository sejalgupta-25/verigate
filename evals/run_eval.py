"""Run the benchmark: for every item, check both the clean and hallucinated
answer against the same evidence, and report catch rate, false-positive
rate, and cost/latency overhead.

Usage:
    OPENAI_API_KEY=... python evals/run_eval.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verigate import Guard, Verdict  # noqa: E402

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def run() -> None:
    items = json.loads(BENCHMARK_PATH.read_text())
    guard = Guard()

    caught, missed = 0, 0
    false_positives, true_negatives = 0, 0
    latencies_ms: list[float] = []
    total_judge_calls, total_prefilter_hits = 0, 0

    rows = []

    for item in items:
        evidence = item["evidence"]

        clean_receipt = guard.check(item["answer_clean"], evidence)
        hallucinated_receipt = guard.check(item["answer_hallucinated"], evidence)

        for receipt in (clean_receipt, hallucinated_receipt):
            latencies_ms.append(receipt.latency_ms)
            total_judge_calls += receipt.judge_calls
            total_prefilter_hits += receipt.prefilter_hits

        clean_flagged = clean_receipt.verdict != Verdict.ALLOW
        hallucinated_flagged = hallucinated_receipt.verdict != Verdict.ALLOW

        if hallucinated_flagged:
            caught += 1
        else:
            missed += 1

        if clean_flagged:
            false_positives += 1
        else:
            true_negatives += 1

        rows.append(
            {
                "id": item["id"],
                "clean_verdict": clean_receipt.verdict.value,
                "hallucinated_verdict": hallucinated_receipt.verdict.value,
            }
        )

    total = len(items)
    catch_rate = caught / total
    false_positive_rate = false_positives / total
    avg_latency = statistics.mean(latencies_ms)

    print(f"\n{'=' * 50}")
    print("Verigate benchmark results")
    print(f"{'=' * 50}")
    print(f"Items:                  {total}")
    print(f"Catch rate:             {catch_rate:.1%}  ({caught}/{total} hallucinated answers flagged)")
    print(f"False-positive rate:    {false_positive_rate:.1%}  ({false_positives}/{total} clean answers wrongly flagged)")
    print(f"Avg latency per check:  {avg_latency:.0f} ms")
    print(f"Judge calls:            {total_judge_calls}  (prefilter resolved {total_prefilter_hits} claims for free)")
    print(f"{'=' * 50}\n")

    for row in rows:
        flag = "OK" if row["hallucinated_verdict"] != "allow" and row["clean_verdict"] == "allow" else "REVIEW"
        print(f"  [{flag:6}] #{row['id']}  clean={row['clean_verdict']:8} hallucinated={row['hallucinated_verdict']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "latest.json"
    out_path.write_text(
        json.dumps(
            {
                "catch_rate": catch_rate,
                "false_positive_rate": false_positive_rate,
                "avg_latency_ms": avg_latency,
                "judge_calls": total_judge_calls,
                "prefilter_hits": total_prefilter_hits,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"Full results written to {out_path}\n")


if __name__ == "__main__":
    run()
