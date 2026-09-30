"""Before/after demo: a toy agent that sometimes embellishes its answer
beyond what the evidence supports, shown raw and then wrapped in @guard.

Usage:
    OPENAI_API_KEY=... python examples/demo_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verigate import guard  # noqa: E402

EVIDENCE = [
    "Acme Corp reported Q2 revenue of $18.4M, up 9% year over year.",
    "The growth was attributed to expansion in the European market.",
]

# Simulates an LLM call that tends to add a confident, unsupported detail.
_FAKE_AGENT_OUTPUT = (
    "Acme Corp's Q2 revenue reached $18.4M, up 9% year over year, "
    "driven by European expansion and a newly signed deal with a Fortune 500 retailer."
)


@guard()
def agent_answer(_question: str) -> str:
    return _FAKE_AGENT_OUTPUT


def main() -> None:
    print("Evidence available to the agent:")
    for e in EVIDENCE:
        print(f"  - {e}")

    print("\nRaw agent output (no verification):")
    print(f"  {_FAKE_AGENT_OUTPUT}\n")

    output_text, receipt = agent_answer("How did Acme Corp do in Q2?", evidence=EVIDENCE)

    print(f"Guarded verdict: {receipt.verdict.value.upper()}")
    for claim in receipt.claims:
        status = "supported" if claim.supported else "UNSUPPORTED"
        print(f"  - [{status}] {claim.claim}  (confidence={claim.confidence:.2f})")

    if receipt.repair_instructions:
        print(f"\nRepair instructions returned to the agent:\n  {receipt.repair_instructions}")


if __name__ == "__main__":
    main()
