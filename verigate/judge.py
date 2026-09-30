"""Claim extraction and evidence verification via an LLM judge.

Two-tier design to keep cost down: a free lexical prefilter disposes of the
claims that are obviously supported or obviously unsupported, and only the
ambiguous middle band pays for a full judge call. A production version would
swap the prefilter for embedding similarity; token overlap is enough to prove
the mechanism without adding an embeddings dependency to a portfolio project.
"""
from __future__ import annotations

import json
import os

from .receipt import ClaimCheck

EXTRACT_PROMPT = """Break the following text into a list of discrete, checkable factual claims.
Return ONLY a JSON array of strings, one per claim. Skip hedges, opinions, and filler.

Text:
{text}
"""

JUDGE_PROMPT = """You are verifying whether a claim is supported by the given evidence.

Claim: {claim}

Evidence:
{evidence}

Respond with ONLY a JSON object:
{{"supported": true/false, "confidence": 0.0-1.0, "evidence_id": "E<n> or null", "reason": "one sentence"}}
"""


class Judge:
    def __init__(self, client=None, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def extract_claims(self, text: str) -> list[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            claims = json.loads(content)
        except json.JSONDecodeError:
            claims = [line.strip("- ") for line in content.splitlines() if line.strip()]
        return [c for c in claims if isinstance(c, str) and c.strip()]

    def prefilter_score(self, claim: str, evidence: list[str]) -> float:
        """Cheap lexical-overlap heuristic; no API call."""
        claim_tokens = _tokenize(claim)
        if not claim_tokens or not evidence:
            return 0.0
        best = 0.0
        for chunk in evidence:
            chunk_tokens = _tokenize(chunk)
            if not chunk_tokens:
                continue
            overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
            best = max(best, overlap)
        return best

    def judge_claim(self, claim: str, evidence: list[str]) -> ClaimCheck:
        evidence_block = "\n".join(f"[E{i}] {chunk}" for i, chunk in enumerate(evidence))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(claim=claim, evidence=evidence_block)}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"supported": False, "confidence": 0.5, "evidence_id": None, "reason": "judge response unparsable"}
        return ClaimCheck(
            claim=claim,
            supported=bool(data.get("supported", False)),
            confidence=float(data.get("confidence", 0.5)),
            evidence_id=data.get("evidence_id"),
            reason=data.get("reason", ""),
        )


def _tokenize(text: str) -> set[str]:
    return {w.strip(".,!?;:\"'()").lower() for w in text.split() if len(w) > 2}
