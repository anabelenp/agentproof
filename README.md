# AgentProof

Enterprise-grade AI agent testing and evaluation framework by Ana Bruno, ThinkAstra Consulting.

AgentProof tests what traditional assertion-based testing cannot: non-deterministic AI outputs across LLM quality, RAG pipelines, multi-model routing, knowledge graph integrity, and governance audit trails.

---

## Implementation Status

| Phase | Component | Status |
|---|---|---|
| Phase 1 | Core infrastructure (config, errors, retry, audit, base, runner) | Complete |
| Phase 2 | LLM evaluation — DeepEval integration, Anthropic SDK | Complete |
| Phase 3 | RAG evaluation — Qdrant retrieval + generation quality | Complete |
| Phase 4 | Routing validation — LiteLLM multi-model routing | Complete |
| Phase 5 | Governance validators, streaming response validation | Complete |
| Phase 6 | Data layer — PostgreSQL + Redis validation | Complete |
| Phase 7 | Graph validation — Neo4j / Memgraph | Not started |
| Phase 8 | Connector validation — Nango ingestion | Not started |
| Phase 9 | Infrastructure validation — Docker, GCP, CI | Not started |
| Phase 10 | CLI (Typer), reporters (Rich + JSON), examples | Not started |

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
│   │   ├── config.py         # AgentProofConfig (Pydantic BaseSettings)
│   │   ├── errors.py         # Exception hierarchy
│   │   ├── retry.py          # retry_async decorator, exponential backoff
│   │   ├── audit.py          # AuditLogger — JSONL + SHA-256 + PII redaction
│   │   ├── safety.py         # PII / prompt-injection detectors
│   │   ├── observability.py  # Prometheus MetricsRegistry + TraceStore
│   │   ├── base.py           # BaseEvaluator, ValidationResult
│   │   └── runner.py         # TestRunner, TestRunSummary
│   ├── evaluators/
│   │   ├── llm.py         # LLMEvaluator (relevance, faithfulness, hallucination, toxicity)
│   │   ├── rag.py         # RAGEvaluator (contextual recall/precision + generation)
│   │   ├── routing.py     # RoutingValidator (correctness, fallback, consistency)
│   │   ├── streaming.py   # StreamingValidator (TTFT, throughput, completeness)
│   │   ├── harness.py     # EvalHarness suite runner
│   │   └── workflow.py    # WorkflowEvaluator (recorded agentic traces)
│   ├── validators/
│   │   ├── governance.py  # GovernanceValidator (trail, override, separation)
│   │   ├── data_layer.py  # DataLayerValidator (cache vs source consistency)
│   │   └── guardrails.py  # PII, injection, policy, tool allowlist
│   └── integrations/
│       ├── anthropic.py   # AnthropicIntegration (async SDK wrapper)
│       ├── qdrant.py      # QdrantIntegration + QdrantEvaluator
│       ├── litellm.py     # LiteLLMIntegration (async acompletion wrapper)
│       ├── postgres.py    # PostgresIntegration + PostgresValidator
│       └── redis.py       # RedisIntegration + RedisValidator
└── tests/
    ├── unit/              # All mocked, fast — runs on every commit
    ├── integration/       # Requires live services — runs on PR merge
    └── regression/        # Canonical test suite — runs every 6 hours
```

## Key design decisions

- **No LangChain, LangGraph, or LangSmith** — ever.
- All evaluator methods are async.
- Every evaluation produces a typed `ValidationResult` (never raw dicts).
- Every result is logged as an immutable JSONL audit entry with SHA-256 hash (PII redacted).
- Prometheus metrics and in-memory traces record pass/fail/latency for every evaluation.
- AgentProof evaluates recorded agentic traces (subagents, skills, MCP tools, background jobs, PR-review gates). It does not host those runtimes.
- `AuditError` always propagates — failed audit writes are compliance violations.

---

**Owner:** Ana Bruno — ThinkAstra Consulting, San Diego CA  
**Target markets:** Insurance, finance, healthcare — enterprise agentic AI QA
