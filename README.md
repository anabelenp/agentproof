# AgentProof

Enterprise-grade AI agent testing and evaluation framework by Ana Bruno, ThinkAstra Consulting.

AgentProof tests what traditional assertion-based testing cannot: non-deterministic AI outputs across LLM quality, RAG pipelines, multi-model routing, knowledge graph integrity, and governance audit trails.

---

## Implementation Status

| Phase | Component | Status |
|---|---|---|
| Phase 1 | Core infrastructure (config, errors, retry, audit, base, runner) | Complete |
| Phase 2 | LLM evaluation — DeepEval integration, Anthropic SDK | Not started |
| Phase 3 | RAG evaluation — Qdrant retrieval + generation quality | Not started |
| Phase 4 | Routing validation — LiteLLM multi-model routing | Not started |
| Phase 5 | Governance validators, streaming response validation | Not started |
| Phase 6 | CLI (Typer), reporters (Rich + JSON), examples | Not started |

---

## Installation

```bash
poetry install
```

## Running tests

```bash
poetry run pytest tests/unit/ -v
```

## Architecture

```
agentproof/
├── src/agentproof/
│   ├── core/
│   │   ├── config.py      # AgentProofConfig (Pydantic BaseSettings)
│   │   ├── errors.py      # Exception hierarchy
│   │   ├── retry.py       # retry_async decorator, exponential backoff
│   │   ├── audit.py       # AuditLogger — JSONL + SHA-256 tamper evidence
│   │   ├── base.py        # BaseEvaluator, ValidationResult
│   │   └── runner.py      # TestRunner, TestRunSummary
│   ├── evaluators/        # Phase 2–5
│   ├── validators/        # Phase 5
│   └── integrations/      # Phase 2–4
└── tests/
    ├── unit/              # All mocked, fast — runs on every commit
    ├── integration/       # Requires live services — runs on PR merge
    └── regression/        # Canonical test suite — runs every 6 hours
```

## Key design decisions

- **No LangChain, LangGraph, or LangSmith** — ever.
- All evaluator methods are async.
- Every evaluation produces a typed `ValidationResult` (never raw dicts).
- Every result is logged as an immutable JSONL audit entry with SHA-256 hash.
- `AuditError` always propagates — failed audit writes are compliance violations.

---

**Owner:** Ana Bruno — ThinkAstra Consulting, San Diego CA  
**Target markets:** Insurance, finance, healthcare — enterprise agentic AI QA
