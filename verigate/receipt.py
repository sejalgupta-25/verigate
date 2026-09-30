"""Structured decision receipts for guarded agent outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"
    REPAIR = "repair"
    BLOCK = "block"


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    confidence: float
    evidence_id: str | None = None
    reason: str = ""


@dataclass
class Receipt:
    verdict: Verdict
    claims: list[ClaimCheck] = field(default_factory=list)
    latency_ms: float = 0.0
    judge_calls: int = 0
    prefilter_hits: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    repair_instructions: str | None = None

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
            "judge_calls": self.judge_calls,
            "prefilter_hits": self.prefilter_hits,
            "claims": [
                {
                    "claim": c.claim,
                    "supported": c.supported,
                    "confidence": round(c.confidence, 3),
                    "evidence_id": c.evidence_id,
                    "reason": c.reason,
                }
                for c in self.claims
            ],
            "repair_instructions": self.repair_instructions,
        }
