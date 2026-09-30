"""Unit tests for verdict logic, using a fake Judge so no API calls are made."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verigate.guard import Guard  # noqa: E402
from verigate.judge import Judge  # noqa: E402
from verigate.receipt import ClaimCheck, Verdict  # noqa: E402


class FakeJudge(Judge):
    """Deterministic stand-in for the real LLM judge."""

    def __init__(self, claims: list[str], verdicts: dict[str, bool]):
        self._claims = claims
        self._verdicts = verdicts

    def extract_claims(self, text: str) -> list[str]:
        return self._claims

    def prefilter_score(self, claim: str, evidence: list[str]) -> float:
        # Force every claim into the ambiguous band so judge_claim always runs.
        return 0.3

    def judge_claim(self, claim: str, evidence: list[str]) -> ClaimCheck:
        supported = self._verdicts.get(claim, False)
        return ClaimCheck(claim=claim, supported=supported, confidence=0.9 if supported else 0.1)


def test_all_supported_allows():
    judge = FakeJudge(claims=["a", "b"], verdicts={"a": True, "b": True})
    guard = Guard(judge=judge)
    receipt = guard.check("a. b.", evidence=["some evidence"])
    assert receipt.verdict == Verdict.ALLOW


def test_partial_unsupported_repairs():
    judge = FakeJudge(claims=["a", "b", "c"], verdicts={"a": True, "b": True, "c": False})
    guard = Guard(judge=judge, block_threshold=0.5)
    receipt = guard.check("a. b. c.", evidence=["some evidence"])
    assert receipt.verdict == Verdict.REPAIR
    assert receipt.repair_instructions is not None
    assert "c" in receipt.repair_instructions


def test_majority_unsupported_blocks():
    judge = FakeJudge(claims=["a", "b", "c"], verdicts={"a": False, "b": False, "c": True})
    guard = Guard(judge=judge, block_threshold=0.5)
    receipt = guard.check("a. b. c.", evidence=["some evidence"])
    assert receipt.verdict == Verdict.BLOCK


def test_prefilter_high_skips_judge_call():
    judge = Judge()
    guard = Guard(judge=judge)
    score = judge.prefilter_score("the sky is blue", ["the sky is blue today"])
    assert score >= guard.prefilter_high


if __name__ == "__main__":
    test_all_supported_allows()
    test_partial_unsupported_repairs()
    test_majority_unsupported_blocks()
    test_prefilter_high_skips_judge_call()
    print("All tests passed.")
