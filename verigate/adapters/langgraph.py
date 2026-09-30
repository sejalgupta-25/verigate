"""LangGraph adapter: verify a node's output before the graph advances.

Framework-agnostic on purpose -- this only assumes LangGraph nodes are
functions of dict state to dict state, so it has no hard dependency on the
langgraph package itself. Drop `guarded_node(...)` in wherever you'd normally
register a node function.
"""
from __future__ import annotations

from typing import Callable

from ..guard import Guard
from ..judge import Judge
from ..receipt import Verdict


def guarded_node(
    node_fn: Callable[[dict], dict],
    output_key: str = "answer",
    evidence_key: str = "evidence",
    receipt_key: str = "receipt",
    max_repairs: int = 1,
    guard: Guard | None = None,
) -> Callable[[dict], dict]:
    """Wrap a LangGraph node so its text output is checked against
    state[evidence_key] before the graph advances.

    On REPAIR, re-invokes node_fn once (up to max_repairs) with repair
    instructions folded into state, mirroring the reflection-loop pattern.
    On BLOCK, leaves the receipt in state for a routing edge to send the
    run to a human-review branch instead of silently failing.
    """
    _guard = guard or Guard(Judge())

    def wrapped(state: dict) -> dict:
        result = node_fn(state)
        evidence = result.get(evidence_key, state.get(evidence_key, []))
        output_text = result.get(output_key, "")
        receipt = _guard.check(output_text, evidence)

        repairs = 0
        while receipt.verdict == Verdict.REPAIR and repairs < max_repairs:
            repair_state = {**state, "repair_instructions": receipt.repair_instructions}
            result = node_fn(repair_state)
            output_text = result.get(output_key, "")
            receipt = _guard.check(output_text, evidence)
            repairs += 1

        result[receipt_key] = receipt.to_dict()
        result["_verigate_verdict"] = receipt.verdict.value
        return result

    return wrapped
