"""guard(): verify LLM-generated claims against evidence before they ship.

The core mechanism: extract atomic claims from an agent's output, check each
one against the evidence it had available, and return allow / repair / block
plus a structured receipt explaining the decision. See receipt.py for the
receipt schema and judge.py for the two-tier scoring.
"""
from __future__ import annotations

import functools
import time
from typing import Callable, Sequence

from .judge import Judge
from .receipt import ClaimCheck, Receipt, Verdict

PREFILTER_LOW = 0.15
PREFILTER_HIGH = 0.55


class Guard:
    def __init__(
        self,
        judge: Judge | None = None,
        block_threshold: float = 0.5,
        prefilter_low: float = PREFILTER_LOW,
        prefilter_high: float = PREFILTER_HIGH,
    ):
        self.judge = judge or Judge()
        self.block_threshold = block_threshold
        self.prefilter_low = prefilter_low
        self.prefilter_high = prefilter_high

    def check(self, output_text: str, evidence: Sequence[str]) -> Receipt:
        start = time.perf_counter()
        claims = self.judge.extract_claims(output_text)
        checks: list[ClaimCheck] = []
        judge_calls = 0
        prefilter_hits = 0

        for claim in claims:
            score = self.judge.prefilter_score(claim, list(evidence))

            if score >= self.prefilter_high:
                checks.append(
                    ClaimCheck(claim=claim, supported=True, confidence=score, reason="prefilter: strong lexical overlap")
                )
                prefilter_hits += 1
                continue

            if score <= self.prefilter_low:
                checks.append(
                    ClaimCheck(claim=claim, supported=False, confidence=1 - score, reason="prefilter: no supporting evidence found")
                )
                prefilter_hits += 1
                continue

            checks.append(self.judge.judge_claim(claim, list(evidence)))
            judge_calls += 1

        unsupported = [c for c in checks if not c.supported]
        if not unsupported:
            verdict = Verdict.ALLOW
        elif len(unsupported) / max(len(checks), 1) > self.block_threshold:
            verdict = Verdict.BLOCK
        else:
            verdict = Verdict.REPAIR

        repair_instructions = None
        if verdict == Verdict.REPAIR:
            repair_instructions = "Revise or remove the following unsupported claims: " + "; ".join(
                c.claim for c in unsupported
            )

        return Receipt(
            verdict=verdict,
            claims=checks,
            latency_ms=(time.perf_counter() - start) * 1000,
            judge_calls=judge_calls,
            prefilter_hits=prefilter_hits,
            repair_instructions=repair_instructions,
        )


def guard(judge: Judge | None = None, **guard_kwargs) -> Callable:
    """Decorator form. Wraps a function returning output text; pass the
    evidence available to it via an `evidence=` kwarg at call time.

    Returns (output_text, receipt) instead of just output_text.
    """
    _guard = Guard(judge, **guard_kwargs)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, evidence: Sequence[str] = (), **kwargs):
            output_text = fn(*args, **kwargs)
            receipt = _guard.check(output_text, evidence)
            return output_text, receipt

        return wrapper

    return decorator
