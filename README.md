# Verigate

A runtime verification layer for agent claims. Wrap an agent's output in
`guard()`, and every claim it makes is checked against the evidence it
actually had before the output ships — allow, repair, or block, with a
structured receipt explaining why.

```
agent output ──▶ extract claims ──▶ check vs. evidence ──▶ verdict + receipt
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼               ▼
                       ALLOW         REPAIR           BLOCK
                  (ships as-is)  (exact fix returned) (sent to review)
```

## Benchmark

Run `python evals/run_eval.py` against the included benchmark (8 items, each
with a clean and a deliberately-embellished answer over the same evidence) to
populate this table:

| Metric | Result |
|---|---|
| Catch rate (hallucinated answers flagged) | _run the eval_ |
| False-positive rate (clean answers wrongly flagged) | _run the eval_ |
| Avg latency per check | _run the eval_ |
| Judge calls vs. prefilter-resolved | _run the eval_ |

False-positive rate matters as much as catch rate: a guard that blocks valid
output gets disabled by whoever has to work around it.

## Quickstart

```python
from verigate import guard

@guard()
def answer_question(question: str) -> str:
    return my_llm_call(question)

output, receipt = answer_question("How did Q2 go?", evidence=[
    "Q2 revenue was $18.4M, up 9% year over year.",
])

if receipt.verdict != "allow":
    print(receipt.repair_instructions)
```

Or wrap a LangGraph node directly:

```python
from verigate.adapters.langgraph import guarded_node

graph.add_node("answer", guarded_node(my_node_fn, output_key="answer", evidence_key="sources"))
```

See [`examples/demo_agent.py`](examples/demo_agent.py) for a full before/after run.

## How it works

1. **Extract claims** — an LLM call breaks the output into discrete,
   checkable statements.
2. **Prefilter** — a free lexical-overlap check disposes of claims that are
   obviously supported or obviously unsupported, so only the ambiguous
   middle band pays for a judge call. (A production version would swap this
   for embedding similarity; token overlap is enough to prove the mechanism
   without adding an embeddings dependency here.)
3. **Judge** — an LLM-as-judge call scores each remaining claim against the
   evidence, with a confidence and a cited evidence ID.
4. **Verdict** — all claims supported → `allow`. Some unsupported → `repair`,
   with exact instructions on what to fix. Majority unsupported → `block`.

## Why I built this

I run a version of this in production at JPMorgan: a reflection loop that
checks a financial research agent's claims against its cited sources before
they reach an analyst, which cut hallucination rate from 18% to 4.2%. That
system is tied to one corpus and one internal pipeline. Verigate is the
generalized version — the same verify-before-ship mechanism, as a small
library that wraps any agent's output rather than one specific system.

## What's not here yet

- **Tool-call / side-effect verification** — checking a proposed action
  against live state (e.g. "is this price still current?") before it
  executes, not just checking a generated claim against static evidence.
  Same mechanism, different guarantee.
- **Embedding-based prefilter** — lexical overlap is a cheap stand-in;
  embeddings would catch paraphrased claims the token-overlap heuristic
  misses.
- **More framework adapters** — CrewAI, AutoGen.
- **Cost/latency tradeoff curve** — sweeping the prefilter thresholds and
  reporting catch rate vs. judge-call volume.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python evals/run_eval.py
python examples/demo_agent.py
```

Offline (no API key needed): `python tests/test_guard.py` runs the verdict
logic against a fake judge.
