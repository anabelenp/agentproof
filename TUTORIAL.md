# AgentProof Tutorial
## From QA Engineer to AI Evaluation Engineer

A guide for people who know Python and test automation but are new to AI output evaluation.

---

## Table of Contents

1. [What Problem Does AgentProof Solve?](#1-what-problem-does-agentproof-solve)
2. [The Tech Stack, Explained](#2-the-tech-stack-explained)
3. [Core Concepts](#3-core-concepts)
4. [Architecture Walkthrough](#4-architecture-walkthrough)
5. [Core Infrastructure Deep Dive (Phase 1)](#5-core-infrastructure-deep-dive-phase-1)
6. [How to Run the Framework](#6-how-to-run-the-framework)
7. [Writing Your First Custom Evaluator](#7-writing-your-first-custom-evaluator)
8. [What Comes Next: Phase 2 Preview](#8-what-comes-next-phase-2-preview)

---

## 1. What Problem Does AgentProof Solve?

Traditional QA is built on a simple premise: for any given input, there is exactly one correct output. You call `calculate_tax(income=50000)` and assert the result equals `7500.00`. The test passes or fails. There is no ambiguity, and no one has to judge whether the answer "feels right." This assumption is so foundational to testing that most testing tools, frameworks, and mental models are built entirely around it.

AI agents break this assumption completely. When a user asks a customer service chatbot "Can I return this after 30 days?", the agent might respond: "Our standard return window is 30 days, so you're right at the boundary — please contact support to check eligibility." Or it might say: "Returns are accepted within 30 days from purchase." Or: "I'd recommend reaching out to our team as your situation is at the limit of our policy." All three responses are semantically correct, grounded in the same knowledge, and equally valid from a customer perspective. But they share not a single common substring. You cannot assert on them. Fuzzy matching and regex patterns both fail here, because correctness is not about characters — it is about meaning.

AgentProof is an evaluation framework built specifically for this gap. Instead of asking "does the output match?", it asks "does the output satisfy a behavioral contract?" — questions like: "is this response faithful to the source material?", "does it answer what was actually asked?", "is the agent hallucinating claims not in the source documents?" These questions require a different kind of evaluator — one that reads and understands the output rather than comparing it to a fixed string. AgentProof provides the infrastructure: typed result contracts, a tamper-evident audit trail, a retry harness for non-deterministic API calls, and a pluggable evaluator pattern that works equally well for LLM quality, RAG retrieval accuracy, multi-model routing consistency, and governance compliance.

---

## 2. The Tech Stack, Explained

### DeepEval

**What it is:** An open-source framework that uses language models as judges to score other language model outputs. It ships with pre-built metrics: answer relevancy, faithfulness, hallucination rate, contextual recall, contextual precision, and toxicity.

**Why AgentProof uses it:** Rather than building LLM-as-judge prompts from scratch, AgentProof delegates metric computation to DeepEval. This keeps evaluator code focused on test orchestration rather than prompt engineering.

**Traditional QA analogy:** Think of DeepEval like a static analysis tool (e.g., SonarQube). You do not read every line of code yourself to check for quality issues — you pass the code to the analyzer and get back a structured quality report. DeepEval does the same for AI outputs.

> ⚠️ **AI TESTING DIFFERENCE:** Unlike SonarQube, DeepEval's scores are not deterministic — the same input can yield slightly different scores across runs because the judge model itself is a language model. This is why AgentProof uses thresholds rather than exact score assertions, and why results must be interpreted as distributions over time rather than point-in-time facts.

---

### LiteLLM

**What it is:** A unified Python interface for calling over 100 different LLM providers — Anthropic, OpenAI, Google, Cohere, Mistral, and more — through a single API.

**Why AgentProof uses it:** Phase 4 validates multi-model routing — testing that an orchestration layer correctly sends legal queries to one model, medical queries to another, and falls back gracefully when a provider is unavailable. LiteLLM makes this testable without requiring separate SDK calls for each provider.

**Traditional QA analogy:** Think of LiteLLM like a database abstraction layer (SQLAlchemy). You write one query in a standard dialect, and the driver handles translation to MySQL, PostgreSQL, or SQLite. LiteLLM does the same for LLM APIs.

---

### Anthropic SDK

**What it is:** Anthropic's official Python SDK for calling Claude models directly — Claude Sonnet, Claude Opus, Claude Haiku.

**Why AgentProof uses it:** The primary judge model (used to score evaluations via DeepEval) is `claude-sonnet-4-6`. The Anthropic SDK is also used to call Claude as the agent under test in LLM evaluation scenarios.

**Traditional QA analogy:** Think of it like a specific JDBC driver for one database. You use it when you need direct, low-level control over the connection and response — not when you need portability across providers.

---

### OpenAI SDK

**What it is:** OpenAI's official Python SDK for calling GPT-4, GPT-4o, and other OpenAI models.

**Why AgentProof uses it:** AgentProof tests multi-model environments. Some organizations run GPT-4 as their primary agent and Claude as a fallback. The OpenAI SDK provides the same direct-control pattern as the Anthropic SDK.

**Traditional QA analogy:** A second JDBC driver, for a different database vendor.

---

### Qdrant

**What it is:** A vector database — stores and queries embeddings (numerical representations of text) rather than traditional rows and columns. Querying it returns documents semantically similar to a search query, not just lexically matching.

**Why AgentProof uses it:** RAG pipelines retrieve context from a vector database before generating a response. Phase 3 tests retrieval quality: did the pipeline retrieve the right documents? Were they relevant? Was anything important missed?

**Traditional QA analogy:** Think of Qdrant like a full-text search engine (Elasticsearch) used as a test fixture store. Instead of matching keywords, it matches meaning — so testing it requires measuring recall and precision rather than exact result set comparison.

---

### Neo4j / Memgraph

**What it is:** A graph database — stores data as nodes and relationships rather than tables and rows. Exceptionally good at representing knowledge graphs, entity networks, and hierarchical relationships.

**Why AgentProof uses it:** Some enterprise AI systems build knowledge graphs from documents, and agents traverse those graphs to answer questions. Phase 7 will validate graph integrity — are relationships correct, complete, and consistent after ingestion?

**Traditional QA analogy:** Schema validation for a relational database, but applied to a graph schema.

---

### PostgreSQL

**What it is:** A battle-tested open-source relational database, used here via `asyncpg` — a high-performance async Python driver.

**Why AgentProof uses it:** Phase 6 tests data layer validators — ensuring that agent-generated outputs persisted to a database meet schema, integrity, and consistency requirements.

**Traditional QA analogy:** Standard database integration testing. Exactly what QA engineers already know.

---

### Redis

**What it is:** An in-memory key-value store used widely for caching, session state, and message queuing.

**Why AgentProof uses it:** AI agents often cache prompt outputs to reduce latency and cost. Phase 6 validates cache correctness — for example, that a cached response is still valid for the same semantic query after a model update.

**Traditional QA analogy:** Cache invalidation testing. The problem is familiar; AgentProof extends it to semantic equivalence rather than exact key matching.

---

### Nango

**What it is:** An open-source platform for building and managing OAuth integrations with external services — Salesforce, HubSpot, Notion, and others.

**Why AgentProof uses it:** Enterprise AI agents often ingest data from third-party connectors. Phase 8 will validate ingestion pipelines — did data from Salesforce land correctly in the vector store? Was it chunked properly? Is it retrievable?

**Traditional QA analogy:** ETL pipeline testing — validating that data arrives correctly and completely from a source system.

---

### FastAPI

**What it is:** A modern Python web framework for building HTTP APIs, built on Pydantic and asyncio.

**Why AgentProof uses it:** If AgentProof is deployed as a service rather than run as a CLI, FastAPI provides the API surface for triggering evaluations, querying results, and streaming audit logs.

**Traditional QA analogy:** A test reporting API — similar to what Allure or TestRail expose, but purpose-built for AI evaluation results.

---

### Pydantic v2

**What it is:** A Python data validation library. You define a class with typed fields, and Pydantic enforces that any data assigned to those fields meets the declared types and constraints — at construction time, not at runtime surprise.

**Why AgentProof uses it:** Every threshold, API key, and configuration value is validated at startup via `AgentProofConfig(BaseSettings)`. Invalid config (e.g., a threshold of `1.5`) raises `ValidationError` before any test runs.

**Traditional QA analogy:** Like a `@BeforeAll` setup validation — if the test environment is misconfigured, fail fast with a clear error message rather than 50 cryptic failures later.

---

### pytest-asyncio

**What it is:** A pytest plugin that lets you write `async def test_...()` functions and run them under pytest's collection and reporting system.

**Why AgentProof uses it:** Every evaluator in AgentProof is async — LLM calls, database queries, file I/O. Without pytest-asyncio, you cannot `await` inside a test function.

**Traditional QA analogy:** Like running Selenium tests in a browser driver context — the framework manages the async lifecycle so you can write straightforward test functions without caring about event loop setup.

---

### Docker

**What it is:** A containerization platform. You package an application and all its dependencies into a container image that runs identically on any machine.

**Why AgentProof uses it:** Phase 9 will validate container health — checking that AI service containers start correctly, respond to health checks, and meet resource constraints.

**Traditional QA analogy:** Smoke tests that run after deployment, but for containers rather than web pages.

---

### GitHub Actions

**What it is:** GitHub's built-in CI/CD system. You define workflows in YAML files; GitHub runs them on push, pull request, or schedule.

**Why AgentProof uses it:** Phase 9 will include a workflow that runs unit tests on every push, integration tests on PR merge, and regression tests on a schedule. **Not yet implemented.**

**Traditional QA analogy:** Jenkins or CircleCI pipelines. The model is identical.

---

### GCP Cloud Run

**What it is:** Google Cloud's serverless container hosting. You deploy a container image; Cloud Run handles scaling, load balancing, and infrastructure.

**Why AgentProof uses it:** Phase 9 will validate Cloud Run deployments — checking that the deployed service is healthy, within latency SLAs, and responding correctly.

**Traditional QA analogy:** Smoke tests against a staging environment, integrated into the deployment pipeline.

---

### Firebase

**What it is:** Google's application platform, commonly used for authentication, real-time databases, and file storage.

**Why AgentProof uses it:** Some enterprise AI applications use Firebase for user management and agent session storage. Phase 9 will validate Firebase connectivity and data correctness.

**Traditional QA analogy:** User management and authentication testing — ensuring signup, login, and token management work correctly.

---

## 3. Core Concepts

### Non-Deterministic Output and Why It Breaks Traditional QA

In traditional software, determinism is the default. Given the same inputs and the same program state, `sum([1, 2, 3])` always returns `6`. Tests rely on this. You run the function once, capture the output, and assert on it forever.

Language models are temperature-sampled probability distributions. The same prompt sent twice may produce the same answer 70% of the time and a slightly different — but equally valid — answer the other 30%. This is not a bug; it is a feature. But it means every assertion like `assert response == expected_str` is fundamentally unreliable as a test strategy.

> ⚠️ **AI TESTING DIFFERENCE:** You cannot record AI responses and replay them like HTTP cassettes (VCR pattern). The recorded response is one valid response at one moment in time. Testing against it indefinitely rewards pattern-matching over quality, and breaks every time the model is updated.

### What Is a Behavioral Contract?

A behavioral contract defines what a response must *do* — not what it must *say*. Instead of:

```python
# Brittle — breaks on any valid paraphrase
assert response == "Our return window is 30 days."
```

A behavioral contract specifies:

```python
# What we actually want to test
assert faithfulness_score(response, source_documents) >= 0.9
assert relevance_score(response, user_query) >= 0.7
assert hallucination_rate(response, source_documents) < 0.1
```

AgentProof implements behavioral contracts as typed `ValidationResult` objects. Each captures a `metric` (what was tested), a `score` (how well it passed), a `threshold` (the pass/fail line), and a `passed` boolean. The contract is the threshold — not the exact score.

### What Is LLM-as-a-Judge?

LLM-as-a-Judge is a technique where a second language model (the "judge") evaluates the output of the first model. You ask the judge: "Given this question and this answer, how relevant is the answer on a scale of 0 to 1?"

This sounds circular, but in practice it works well because the judge evaluates semantic quality — not string equality. The judge reasons: "The answer addresses the question, provides correct policy information, and does not add false claims. Score: 0.87."

AgentProof uses DeepEval to run this pattern. The default judge model is `claude-sonnet-4-6`, configurable via the `JUDGE_MODEL` environment variable.

> ⚠️ **AI TESTING DIFFERENCE:** LLM-as-a-Judge introduces evaluation cost (API calls to the judge model) and latency. This is fundamentally different from traditional assertion-based testing, which is instantaneous and free. Budget for judge-model calls when designing your evaluation suite.

### What Is a RAG Pipeline and Why Retrieval Quality Matters?

RAG (Retrieval-Augmented Generation) is the pattern where an AI agent, before generating a response, retrieves relevant documents from a knowledge base (usually a vector database) and includes them in its context window. This grounds the response in actual source material rather than the model's parametric memory.

Retrieval quality matters because the agent can only answer as well as the documents it retrieves. If the retrieval step returns irrelevant or incomplete documents, even a perfect generation step cannot compensate. This is why Phase 3 tests the retrieval and generation stages independently.

> ⚠️ **AI TESTING DIFFERENCE:** In traditional QA, test data is exact — you query a database and get specific rows. In RAG, retrieval is approximate and ranked by semantic similarity. Testing it requires measuring recall (did we get the right documents?) and precision (did we get only relevant documents?), not exact result set comparison.

### What Is Prompt Regression Testing?

Prompt regression testing runs a canonical set of evaluations against a fixed set of prompts and responses every time a prompt is changed, a model is upgraded, or a retrieval pipeline is modified.

Traditional regression testing catches functional regressions: "did this code change break that feature?" Prompt regression testing catches quality regressions: "did upgrading from GPT-4 to GPT-4o cause our customer service bot to become less faithful to policy documents?"

### What Is Multi-Model Routing and Why Does Consistency Matter?

Multi-model routing is the pattern where different queries are sent to different models based on cost, latency, capability, or compliance requirements — for example, routing queries containing PII to an on-premises model, and general queries to a cloud API.

Testing routing consistency means verifying that: (a) the router sends the right query to the right model, (b) the response quality is acceptable regardless of which model was selected, and (c) fallback behavior works correctly when a model is unavailable.

> ⚠️ **AI TESTING DIFFERENCE:** Model routing is invisible in traditional QA — the database driver abstraction hides which backend executes a query. In AI systems, the routing decision directly affects output quality, cost, latency, and compliance. It must be explicitly tested.

---

## 4. Architecture Walkthrough

### Folder Structure

```
agentproof/
├── src/agentproof/
│   ├── core/                    # Phase 1 — COMPLETE AND TESTED
│   │   ├── config.py            # AgentProofConfig: Pydantic BaseSettings
│   │   ├── errors.py            # Exception hierarchy
│   │   ├── retry.py             # retry_async decorator + RetryConfig
│   │   ├── audit.py             # AuditLogger: JSONL + SHA-256 + PII redaction
│   │   ├── safety.py            # PII / prompt-injection detectors (COMPLETE)
│   │   ├── observability.py     # Prometheus MetricsRegistry + TraceStore (COMPLETE)
│   │   ├── base.py              # BaseEvaluator (abstract) + ValidationResult
│   │   └── runner.py            # TestRunner + TestRunSummary
│   ├── integrations/            # Phases 2–7 complete; 8–9 not started
│   │   ├── anthropic.py         # Async Anthropic SDK wrapper (Phase 2 — COMPLETE)
│   │   ├── litellm.py           # LiteLLM routing wrapper (Phase 4 — COMPLETE)
│   │   ├── qdrant.py            # QdrantIntegration + QdrantEvaluator (Phase 3 — COMPLETE)
│   │   ├── postgres.py          # PostgresIntegration + PostgresValidator (Phase 6 — COMPLETE)
│   │   ├── redis.py             # RedisIntegration + RedisValidator (Phase 6 — COMPLETE)
│   │   ├── neo4j.py             # Neo4jIntegration async Bolt wrapper (Phase 7 — COMPLETE)
│   │   ├── nango.py             # Nango connector client (Phase 8)
│   │   ├── docker.py            # Docker SDK wrapper (Phase 9)
│   │   └── gcp.py               # GCP Cloud Run client (Phase 9)
│   ├── evaluators/              # Phases 2–5 complete + harness/workflow
│   │   ├── llm.py               # LLMEvaluator: relevance, faithfulness, hallucination, toxicity (COMPLETE)
│   │   ├── rag.py               # RAGEvaluator: recall, precision, generation (COMPLETE)
│   │   ├── routing.py           # RoutingValidator: correctness, fallback, consistency (COMPLETE)
│   │   ├── streaming.py         # StreamingValidator: TTFT, throughput (COMPLETE)
│   │   ├── harness.py           # EvalHarness + EvalCase suite runner (COMPLETE)
│   │   └── workflow.py          # WorkflowEvaluator: recorded agentic traces (COMPLETE)
│   ├── validators/              # Phases 5–7 complete + guardrails; 8 not started
│   │   ├── governance.py        # GovernanceValidator: audit trail completeness (COMPLETE)
│   │   ├── data_layer.py        # DataLayerValidator: cache vs source consistency (COMPLETE)
│   │   ├── guardrails.py        # PII, injection, policy, tool allowlist (COMPLETE)
│   │   └── graph.py             # GraphValidator: entity resolution, relationships (COMPLETE)
│   └── cli.py                   # Phase 10 — NOT YET IMPLEMENTED
├── tests/
│   ├── unit/                    # Phases 1–7 + evals/guardrails/observability — mocked, run on every commit
│   ├── integration/             # Future — requires live services
│   └── regression/              # Future — canonical suite, runs on schedule
├── audit_logs/                  # Created at runtime — not committed to git
│   └── audit_YYYY-MM-DD.jsonl  # One file per UTC day
├── pyproject.toml
├── .env.example
└── CLAUDE.md
```

**Why each layer exists:**

- `core/` is the plumbing. It knows nothing about LLMs or databases — only about results, audit logging, retry behavior, error types, PII redaction, and Prometheus traces. This isolation is deliberate: if DeepEval is swapped for a different evaluation library, the core layer does not change.
- `integrations/` wraps external SDKs. It translates between AgentProof's internal types and the external API's types. Keeping SDK calls here means evaluators stay readable and mockable.
- `evaluators/` implements specific behavioral contracts using the integration layer. An `LLMEvaluator` uses `AnthropicIntegration` to call the model and DeepEval to score the result. `WorkflowEvaluator` scores a *recorded* agentic trace — it does not host subagents, skills, or MCP servers.
- `validators/` is for infrastructure-level checks (governance, data layer, guardrails) that do not fit the "score a model output" pattern of evaluators.
- `tests/unit/` mocks every external dependency — no API keys or running services required. This is the suite that runs on every commit.

### Data Flow: Test Trigger to Audit Log Entry

```
┌────────────────────────────────────────────────────────────────────┐
│                      TEST TRIGGER                                  │
│   pytest / CLI / CI system calls runner.run()                      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                     TestRunner.run()                               │
│                                                                    │
│  parallel=True:   asyncio.gather([ev.evaluate() for each ev])     │
│  parallel=False:  sequential for-loop in registration order       │
└──────────┬──────────────────────────────────────┬─────────────────┘
           │                                      │
           ▼                                      ▼
┌────────────────────────┐            ┌────────────────────────────┐
│     Evaluator A        │            │     Evaluator B            │
│                        │            │                            │
│ 1. _new_audit_id()     │            │ 1. _new_audit_id()         │
│ 2. _start_timer()      │            │ 2. _start_timer()          │
│ 3. score the output    │            │ 3. score the output        │
│    (deepeval / API)    │            │    (deepeval / API)        │
│ 4. build result        │            │ 4. build result            │
│ 5. _write_audit(result)│            │ 5. _write_audit(result)    │
└───────────┬────────────┘            └───────────┬────────────────┘
            │                                     │
            └─────────────────┬───────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                    AuditLogger.log()                               │
│                                                                    │
│  1. Set entry.timestamp  (ISO-8601 UTC)                            │
│  2. Compute SHA-256 hash over all entry fields (excl. hash field)  │
│  3. Acquire asyncio.Lock  (serializes concurrent writers)          │
│  4. Append one JSONL line via asyncio.to_thread()                  │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│              audit_logs/audit_2026-05-25.jsonl                     │
│                                                                    │
│  {"audit_id":"550e8400-...","evaluator":"LLMEvaluator",           │
│   "metric":"faithfulness","score":0.94,"threshold":0.9,           │
│   "passed":true,"model":"claude-sonnet-4-6","latency_ms":342.1,   │
│   "timestamp":"2026-05-25T14:30:00Z","entry_hash":"sha256:a3f2…"} │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                     TestRunSummary                                 │
│                                                                    │
│  run_id="...", total=2, passed=1, failed=1, error_count=0         │
│  pass_rate=0.5, all_passed=False                                   │
│  results=[ValidationResult, ValidationResult]                      │
└────────────────────────────────────────────────────────────────────┘
```

### Evaluator Class Hierarchy

```
ABC  (Python abstract base class)
└── BaseEvaluator  (abstract — src/agentproof/core/base.py)
    │
    │  Implements: __init__, _new_audit_id, _start_timer,
    │              _elapsed_ms, _error_result, _write_audit
    │  Requires subclass to implement: name (property), evaluate()
    │
    ├── LLMEvaluator       (Phase 2 — evaluators/llm.py)      COMPLETE AND TESTED
    ├── RAGEvaluator        (Phase 3 — evaluators/rag.py)      COMPLETE AND TESTED
    ├── RoutingValidator    (Phase 4 — evaluators/routing.py)  COMPLETE AND TESTED
    ├── StreamingValidator  (Phase 5 — evaluators/streaming.py) COMPLETE AND TESTED
    ├── GovernanceValidator (Phase 5 — validators/governance.py) COMPLETE AND TESTED
    └── [YourCustomEvaluator]  (see Section 7 for a complete example)
```

### CI/CD Pipeline Trigger Matrix (Planned — Phase 9)

> **Note:** `.github/workflows/` does not exist yet. This diagram represents the intended design to be implemented in Phase 9.

```
┌──────────────────┬────────────────┬────────────────┬─────────────────┐
│ Trigger          │ Unit Tests     │ Integration    │ Regression Suite│
│                  │ (no live deps) │ Tests          │ (canonical set) │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ git push         │ ✓ RUNS         │ ✗ SKIPPED      │ ✗ SKIPPED       │
│ (any branch)     │                │                │                 │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ pull request     │ ✓ RUNS         │ ✓ RUNS         │ ✗ SKIPPED       │
│ (to main)        │                │                │                 │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ merge to main    │ ✓ RUNS         │ ✓ RUNS         │ ✓ RUNS          │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ schedule         │ ✗ SKIPPED      │ ✗ SKIPPED      │ ✓ RUNS          │
│ (every 6 hours)  │                │                │                 │
└──────────────────┴────────────────┴────────────────┴─────────────────┘
```

---

## 5. Core Infrastructure Deep Dive (Phase 1)

This section documents exactly what is implemented and tested as of Phase 1. All code shown here runs today with `poetry run pytest tests/unit/ -v`.

### AgentProofConfig

`AgentProofConfig` is a Pydantic `BaseSettings` class. It loads configuration from environment variables and a `.env` file, validates all values at construction time, and provides typed access to every configuration setting.

**How Pydantic BaseSettings works:** Field names map to uppercase environment variables automatically. `anthropic_api_key` maps to `ANTHROPIC_API_KEY`. `qdrant_url` maps to `QDRANT_URL`. Case is not sensitive. If a field is not set in the environment, the declared default is used. If a field with a validator (like `ge=0.0, le=1.0`) receives an out-of-range value, Pydantic raises `ValidationError` immediately — before any evaluation runs.

**Your `.env` file:**

```bash
# Copy .env.example to .env and fill in your values.
# Never commit .env to git.

ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
QDRANT_API_KEY=your-qdrant-cloud-key
QDRANT_URL=http://localhost:6333
AUDIT_LOG_DIR=./audit_logs

# Optional overrides — these have working defaults
# JUDGE_MODEL=claude-sonnet-4-6
# MAX_RETRIES=3
```

**Using it in code:**

```python
from agentproof.core.config import AgentProofConfig

# Reads from .env automatically
config = AgentProofConfig()

# Override a field programmatically (useful in tests)
config = AgentProofConfig(default_relevance_threshold=0.85)

# Access values
print(config.judge_model)                # "claude-sonnet-4-6"
print(config.faithfulness_threshold)     # 0.9
print(config.anthropic_api_key)          # None (if not set in env)
```

**Complete field reference:**

| Field | Default | Constraints | Description |
|---|---|---|---|
| `anthropic_api_key` | `None` | — | Claude API key |
| `openai_api_key` | `None` | — | OpenAI API key |
| `qdrant_api_key` | `None` | — | Qdrant Cloud API key |
| `judge_model` | `"claude-sonnet-4-6"` | — | Model used for LLM-as-a-Judge |
| `qdrant_url` | `"http://localhost:6333"` | — | Qdrant server URL |
| `audit_log_dir` | `Path("./audit_logs")` | — | Directory for JSONL log files |
| `default_relevance_threshold` | `0.7` | `[0.0, 1.0]` | Minimum relevance score |
| `faithfulness_threshold` | `0.9` | `[0.0, 1.0]` | Minimum faithfulness score |
| `executive_faithfulness_threshold` | `0.95` | `[0.0, 1.0]` | For executive-facing outputs |
| `hallucination_threshold` | `0.1` | `[0.0, 1.0]` | Maximum hallucination rate |
| `contextual_recall_threshold` | `0.7` | `[0.0, 1.0]` | Minimum RAG recall |
| `contextual_precision_threshold` | `0.8` | `[0.0, 1.0]` | Minimum RAG precision |
| `max_ttft_seconds` | `2.0` | `> 0` | Max time-to-first-token SLA |
| `min_token_throughput` | `20.0` | `> 0` | Min tokens/second SLA |
| `max_retries` | `3` | `[1, 10]` | Max retry attempts |
| `retry_base_delay` | `1.0` | `> 0` | Base delay in seconds |
| `retry_max_delay` | `60.0` | `> 0` | Maximum delay cap |

---

### ValidationResult

`ValidationResult` is the canonical return type for every evaluator. Think of it like a `TestResult` object in JUnit — it captures whether the test passed, the score, and diagnostic information. Unlike JUnit's binary pass/fail, a `ValidationResult` has a continuous score from 0.0 to 1.0 and a configurable threshold.

> ⚠️ **AI TESTING DIFFERENCE:** In JUnit, a test either passes or fails — binary. In AgentProof, a result has a continuous score, and the pass/fail decision is made by comparing that score to a configurable threshold. The same score of 0.75 can be a "pass" at threshold 0.7 and a "fail" at threshold 0.85. This is intentional: different use cases (executive reporting vs. internal QA) may require different quality bars for the same metric.

**Why typed dataclasses over raw dicts:** Returning a raw `dict` from an evaluator is explicitly forbidden by AgentProof's design. The reason is that dicts have no guaranteed keys, no type enforcement, and no validation. A `ValidationResult` dataclass guarantees — at construction time via `__post_init__` — that `score` is between 0.0 and 1.0 and every field is present and typed.

**Field reference:**

```python
@dataclass
class ValidationResult:
    passed: bool          # True if score >= threshold
    score: float          # 0.0–1.0 (enforced: ValueError if outside range)
    evaluator_name: str   # matches BaseEvaluator.name
    metric: str           # "faithfulness", "relevance", "hallucination", etc.
    threshold: float      # the pass/fail boundary used for this evaluation
    details: dict[str, Any]  # metric-specific context — reasoning, sub-scores
    latency_ms: float     # wall-clock evaluation time in milliseconds
    timestamp: datetime   # UTC datetime when evaluation completed
    audit_id: str         # UUID linking this result to its AuditEntry
    error: str | None     # set on failure; None on success — never raise instead
```

**Reading a result:**

```python
result = await evaluator.evaluate(input="What is the refund policy?",
                                  output="Refunds are available for 30 days.")

if result.error:
    print(f"ERROR [{result.evaluator_name}] {result.metric}: {result.error}")
    print(f"  Audit reference: {result.audit_id}")
elif result.passed:
    print(f"PASS  [{result.evaluator_name}] {result.metric}: "
          f"{result.score:.2f} >= {result.threshold}")
else:
    print(f"FAIL  [{result.evaluator_name}] {result.metric}: "
          f"{result.score:.2f} < {result.threshold}")
    print(f"  Reason: {result.details.get('reason', 'no details')}")
```

---

### BaseEvaluator

`BaseEvaluator` is the abstract base class every evaluator inherits from. It enforces a strict contract via Python's `ABC` mechanism: any subclass that does not implement `name` and `evaluate()` cannot be instantiated — Python raises `TypeError` at construction time.

**The two things every subclass must implement:**

1. `name` (property returning `str`) — a unique identifier used in audit logs and test reports.
2. `evaluate(**kwargs)` (async method returning `ValidationResult`) — runs the evaluation.

**The contract `evaluate()` must follow:**

```python
async def evaluate(self, **kwargs) -> ValidationResult:
    # Step 1 — generate the shared ID before anything happens
    audit_id = self._new_audit_id()

    # Step 2 — start timing
    start = self._start_timer()

    try:
        # Step 3 — do the actual evaluation work
        score = await self._compute_score(...)

        # Step 4 — build the typed result
        result = ValidationResult(
            passed=score >= threshold,
            score=score,
            evaluator_name=self.name,
            metric="my_metric",
            threshold=threshold,
            details={"reason": "..."},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
        )
    except Exception as exc:
        # Step 4 (failure path) — convert to error result, do NOT raise
        result = self._error_result(audit_id, start, "my_metric", exc, threshold)

    # Step 5 — ALWAYS write audit before returning
    await self._write_audit(result)
    return result
```

**Helper methods provided by BaseEvaluator — you do not implement these:**

| Method | What it does |
|---|---|
| `_new_audit_id()` | Returns `str(uuid.uuid4())` — generates the shared UUID |
| `_start_timer()` | Returns `time.perf_counter()` — high-resolution start time |
| `_elapsed_ms(start)` | Returns `(perf_counter() - start) * 1000.0` in milliseconds |
| `_error_result(audit_id, start, metric, exc, threshold)` | Builds a failed `ValidationResult` from an exception — `passed=False`, `score=0.0`, `error` set |
| `_write_audit(result, model="")` | Constructs an `AuditEntry` from the result and calls `self._audit.log(entry)` |

> ⚠️ **AI TESTING DIFFERENCE:** In traditional test frameworks, a test that raises an exception is automatically marked as an error and the runner moves on. In AgentProof, evaluators must catch their own exceptions and convert them to error `ValidationResult` objects. This is because the audit trail must record every evaluation attempt — including failed ones — and the `TestRunner` needs a typed result to aggregate, even for failures.

---

### AuditLogger

`AuditLogger` writes a tamper-evident audit trail of every evaluation result. Each entry is a single JSON line appended to a daily log file. Each line includes a SHA-256 hash over all other fields, making retroactive modification detectable.

**Why tamper-evident logging matters in AI systems:**

In regulated industries — insurance, finance, healthcare — AI-generated decisions may be subject to regulatory review. An auditor needs to verify that "the model scored 0.94 on faithfulness on 2026-05-25" was the actual recorded result at the time of evaluation, not a number changed retroactively. The SHA-256 hash over each record provides this guarantee.

**A real audit log entry:**

```json
{
  "audit_id": "550e8400-e29b-41d4-a716-446655440000",
  "evaluator": "LLMEvaluator",
  "metric": "faithfulness",
  "score": 0.94,
  "threshold": 0.9,
  "passed": true,
  "model": "claude-sonnet-4-6",
  "latency_ms": 342.1,
  "details": {"reason": "Output is grounded in the provided context"},
  "error": null,
  "timestamp": "2026-05-25T14:30:00.123456+00:00",
  "entry_hash": "sha256:a3f2c1e4b9d7e6f8a2b4c6d8e0f2a4b6c8d0e2f4a6b8c0d2e4f6a8b0c2d4e6f8"
}
```

**How the hash works:** The `entry_hash` field is computed as `sha256(json.dumps(all_other_fields, sort_keys=True))`. Changing any other field — even flipping `passed` from `true` to `false`, or altering the score by 0.01 — produces a completely different hash. `verify_integrity(audit_id)` re-reads the file, re-computes the hash, and returns `False` if they do not match.

**How the audit_id links results to log entries:** The `audit_id` UUID is generated by the evaluator (via `self._new_audit_id()`) *before* the evaluation runs. It is embedded in both the returned `ValidationResult` and the `AuditEntry`. Given any result from a test run, you can look up its corresponding audit log entry by `audit_id` — they always match.

**Concurrency safety:** Multiple evaluators running in parallel via `asyncio.gather` share the same `AuditLogger`. The logger uses `asyncio.Lock` to serialize writes — only one coroutine appends at a time. File I/O runs inside `asyncio.to_thread()` so it never blocks the event loop.

---

### RetryConfig and retry_async

`retry_async` is a decorator that wraps any async function with exponential backoff retry logic. All LLM API calls and external service calls in AgentProof use this pattern to handle transient failures — rate limits, network timeouts, 5xx errors.

**Why exponential backoff with jitter?**

When an API returns a rate-limit error (HTTP 429), every client that hit the limit at the same moment will retry at the same moment — causing a thundering herd that re-triggers the rate limit. Jitter adds a random multiplier (0.5–1.5) to each computed delay, spreading retries across time.

**A concrete backoff example:**

```
Config: max_attempts=4, base_delay=1.0, exponential_base=2.0, jitter=False

Attempt 1 → fails → sleep 1.0 * 2^0 = 1.0s
Attempt 2 → fails → sleep 1.0 * 2^1 = 2.0s
Attempt 3 → fails → sleep 1.0 * 2^2 = 4.0s
Attempt 4 → fails → raise RetryExhaustedError (last_exception=<the cause>)

With jitter=True, each sleep is multiplied by random(0.5, 1.5):
Attempt 1 → sleep 0.5s–1.5s
Attempt 2 → sleep 1.0s–3.0s
Attempt 3 → sleep 2.0s–6.0s
```

**Using the decorator:**

```python
from agentproof.core.retry import RetryConfig, retry_async
from agentproof.core.errors import RetryExhaustedError

my_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(ConnectionError, TimeoutError),
    # Only these exception types trigger a retry.
    # A 401 Unauthorized propagates immediately — no retry.
)

@retry_async(my_config)
async def call_external_api(prompt: str) -> str:
    response = await some_api_client.generate(prompt)
    return response.text

try:
    result = await call_external_api("summarize this document")
except RetryExhaustedError as e:
    print(f"All retries exhausted. Last error: {e.last_exception}")
```

**Exception filtering:** The `retryable_exceptions` parameter controls which exception types trigger retries. If the function raises a type not in the tuple, it propagates immediately without any retry. This is important: if a cloud API returns `401 Unauthorized`, retrying is wasteful — fail fast.

---

### Error Hierarchy

AgentProof uses a typed exception hierarchy rather than catching bare `Exception`. Each class carries different semantics and different handling behavior.

```
Exception
└── AgentProofError
    ├── ConfigurationError      — bad config at startup; fix before running
    ├── EvaluatorError          — evaluator execution failure
    │   ├── LLMEvaluatorError
    │   ├── RAGEvaluatorError
    │   ├── RoutingValidatorError
    │   ├── StreamingValidatorError
    │   └── GovernanceValidatorError
    ├── AuditError              — ALWAYS propagates — compliance violation
    ├── RetryExhaustedError     — all retry attempts exhausted
    └── IntegrationError        — external service failure
```

**The fail-gracefully pattern:** `EvaluatorError` and its subclasses are never allowed to propagate out of `evaluate()`. They are caught inside the method and converted to error `ValidationResult` objects with `error` set. The pipeline continues. The failure is recorded in the audit log.

**Why `AuditError` is different:** `AuditError` always propagates and is never swallowed. A failed audit write does not mean the evaluation had a bad result — it means there is *no record* of the result. In regulated industries, this is categorically different from a low score or an evaluator error. It is a missing compliance record and must surface immediately.

> ⚠️ **AI TESTING DIFFERENCE:** Traditional test frameworks treat all internal errors the same — they appear as test errors in the report. AgentProof distinguishes three states explicitly: *passed* (behavioral contract met), *failed* (behavioral contract not met — `error=None`), and *errored* (evaluation could not complete — `error` is set). This distinction matters for SLA reporting and compliance dashboards.

---

### TestRunner

`TestRunner` orchestrates multiple evaluators into a single test run, collecting all results into a `TestRunSummary`.

**The registration pattern:**

```python
from agentproof.core.config import AgentProofConfig
from agentproof.core.runner import TestRunner
from agentproof.core.audit import AuditLogger

config = AgentProofConfig()
runner = TestRunner(config)

# Register evaluators with the kwargs they need at evaluate() time.
# These are not executed yet — just registered.
runner.register(evaluator_a, input="user query", output="model response")
runner.register(evaluator_b, query="search query", documents=["doc1", "doc2"])

# Now execute all of them
summary = await runner.run(parallel=True)   # default: concurrent
summary = await runner.run(parallel=False)  # or: sequential, in order
```

**`TestRunner` creates its own `AuditLogger`** from `config.audit_log_dir`. All evaluators registered to the same runner write to the same daily JSONL file — no separate logger setup required.

**Parallel vs sequential:**

- `parallel=True` (default): runs all evaluators concurrently via `asyncio.gather`. Total time ≈ time of the slowest evaluator. Use for independent evaluators.
- `parallel=False`: runs evaluators in registration order. Use when evaluator B's input depends on evaluator A's output.

**Error isolation:** `TestRunner._run_one()` wraps each `evaluate()` call in a safety-net try/except. If an evaluator has a bug that causes it to raise rather than return an error result, `_run_one` catches it, converts it to an error `ValidationResult`, and the run continues. One broken evaluator never kills the rest.

**Reading the summary:**

```python
summary = await runner.run()

print(f"Total evaluators: {summary.total}")
print(f"Passed:           {summary.passed}")
print(f"Failed:           {summary.failed}")
print(f"Errors:           {summary.error_count}")
print(f"Pass rate:        {summary.pass_rate:.1%}")
print(f"All passed:       {summary.all_passed}")
print(f"Duration:         {summary.duration_ms:.0f}ms")

# Standard CI exit code pattern
import sys
sys.exit(0 if summary.all_passed else 1)
```

**`all_passed` is strict:** It is `True` only when `failed == 0` AND `error_count == 0`. A run with zero failures but one error is not `all_passed`. This matches the expectation for regulated environments: errors are not acceptable, not just tallied.

---

## 6. How to Run the Framework

### Prerequisites

- Python 3.11 — the framework requires 3.11 specifically (3.13 causes grpcio compile failures with deepeval)
- [Poetry](https://python-poetry.org/) — package and virtual environment manager

If you use conda:

```bash
conda create -n agentproof python=3.11
conda activate agentproof
```

### Installation

```bash
# Clone the repository
git clone https://github.com/anabelenp/agentproof
cd agentproof

# Install all dependencies (including dev tools)
poetry install

# If Poetry picks up the wrong Python version, point it explicitly:
poetry env use /path/to/python3.11
poetry install

# Configure environment
cp .env.example .env
# Open .env in your editor and add API keys
```

### External Services for Unit Tests

Unit tests require **no external services**. Every dependency — Anthropic API, Qdrant, databases — is mocked via `unittest.mock`. You can run the full unit test suite with no API keys and no running containers.

A `docker-compose.yml` for local development (Qdrant, PostgreSQL, Redis) is planned for Phase 9 and does not exist yet.

### Running Unit Tests

```bash
poetry run pytest tests/unit/ -v
```

Expected output (all 91 tests pass in under one second):

```
============================== test session starts ==============================
platform darwin -- Python 3.11.x, pytest-8.x, asyncio_mode=auto
collected 91 items

tests/unit/test_audit.py::test_log_creates_directory_and_file PASSED
tests/unit/test_audit.py::test_log_writes_valid_jsonl PASSED
tests/unit/test_audit.py::test_log_includes_sha256_hash PASSED
tests/unit/test_audit.py::test_verify_integrity_valid_entry PASSED
tests/unit/test_audit.py::test_verify_integrity_tampered_score PASSED
tests/unit/test_audit.py::test_verify_integrity_tampered_passed_flag PASSED
tests/unit/test_base.py::test_evaluate_returns_validation_result PASSED
tests/unit/test_base.py::test_audit_id_in_result_matches_audit_entry PASSED
tests/unit/test_base.py::test_error_result_structure PASSED
tests/unit/test_config.py::test_default_thresholds PASSED
tests/unit/test_config.py::test_threshold_too_high_raises PASSED
tests/unit/test_config.py::test_env_var_sets_api_key PASSED
tests/unit/test_errors.py::test_audit_error_not_caught_as_evaluator_error PASSED
tests/unit/test_retry.py::test_exponential_backoff_without_jitter PASSED
tests/unit/test_retry.py::test_does_not_retry_non_configured_exception PASSED
tests/unit/test_runner.py::test_one_error_does_not_abort_others PASSED
tests/unit/test_runner.py::test_error_field_set_counts_as_error_not_failed PASSED
... (74 more tests)

========================= 91 passed in 0.71s ============================
```

### What Passing vs Failing vs Error Results Look Like

**A passing result (score above threshold):**
```
PASS  [LLMEvaluator] faithfulness: 0.94 >= 0.90
      Reason: Output is grounded in the provided context documents.
      Latency: 342ms
      Audit ID: 550e8400-e29b-41d4-a716-446655440000
```

**A failing result (score below threshold):**
```
FAIL  [LLMEvaluator] faithfulness: 0.62 < 0.90
      Reason: Output contains claims not found in source documents.
      Latency: 289ms
      Audit ID: 660e8400-f39b-52e5-b827-557766551111
```

**An error result (evaluation could not complete):**
```
ERROR [LLMEvaluator] faithfulness: TimeoutError: API did not respond within 30s
      Score: 0.00  Passed: False
      Audit ID: 770e8400-g49b-63f6-c938-668877662222
```

Note that errors and failures are distinct: a failure means the evaluation ran and the output did not meet the threshold. An error means the evaluation itself could not complete. Both are recorded in the audit log. Both contribute to `all_passed=False`. But they are counted separately in `failed` vs `error_count`.

---

## 7. Writing Your First Custom Evaluator

This section walks through building a complete, working evaluator that checks whether a model response stays within a character length limit. This is a simple but real use case: enterprise chatbots deployed in helpdesk or call center contexts often have strict output length requirements imposed by the UI or operator.

### Step 1: Define the Behavioral Contract

The contract: "A valid response must not exceed `max_chars` characters." We model this as a continuous score rather than a binary check, because a response 5 characters over the limit is meaningfully different from one 500 characters over.

Scoring logic:
- `score = 1.0` if `len(response) <= max_chars`
- `score = 0.0` if `len(response) >= 2 * max_chars`
- `score` decreases linearly between those bounds

### Step 2: Create the Evaluator

Create `src/agentproof/evaluators/conciseness.py`:

```python
"""ConcisenessEvaluator — checks that model responses stay within a character limit.

Scores 1.0 for responses within the limit, decreasing linearly to 0.0
for responses at or beyond twice the limit.
"""

from datetime import datetime, timezone

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig


class ConcisenessEvaluator(BaseEvaluator):
    """Evaluates whether a model response is within a character length constraint.

    Args:
        config: AgentProofConfig instance.
        audit_logger: AuditLogger for tamper-evident result recording.
        max_chars: Maximum allowed character count. Default: 500.
        threshold: Minimum score to pass. Default: 0.8.

    Usage:
        evaluator = ConcisenessEvaluator(config, audit_logger, max_chars=300)
        result = await evaluator.evaluate(response="The answer is 42.")
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        max_chars: int = 500,
        threshold: float = 0.8,
    ) -> None:
        super().__init__(config, audit_logger)
        self._max_chars = max_chars
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs."""
        return "conciseness_evaluator"

    async def evaluate(self, *, response: str, **kwargs: object) -> ValidationResult:
        """Evaluate whether the response meets the character length constraint.

        Args:
            response: The model-generated text to evaluate.

        Returns:
            ValidationResult with score=1.0 if within limit, lower if over.
            On unexpected failure, returns an error result with passed=False.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()

        try:
            char_count = len(response)

            if char_count <= self._max_chars:
                score = 1.0
            elif char_count >= self._max_chars * 2:
                score = 0.0
            else:
                overage = char_count - self._max_chars
                score = 1.0 - (overage / self._max_chars)

            result = ValidationResult(
                passed=score >= self._threshold,
                score=round(score, 4),
                evaluator_name=self.name,
                metric="conciseness",
                threshold=self._threshold,
                details={
                    "char_count": char_count,
                    "max_chars": self._max_chars,
                    "over_by": max(0, char_count - self._max_chars),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )

        except Exception as exc:
            result = self._error_result(audit_id, start, "conciseness", exc, self._threshold)

        await self._write_audit(result)
        return result
```

### Step 3: Write Tests

Create `tests/unit/test_conciseness_evaluator.py`:

```python
"""Unit tests for ConcisenessEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.base import ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.evaluators.conciseness import ConcisenessEvaluator


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(mock_audit) -> ConcisenessEvaluator:
    return ConcisenessEvaluator(AgentProofConfig(), mock_audit, max_chars=100, threshold=0.8)


async def test_short_response_scores_one(evaluator):
    result = await evaluator.evaluate(response="Short answer.")
    assert result.score == 1.0
    assert result.passed is True


async def test_response_at_exact_limit_scores_one(evaluator):
    result = await evaluator.evaluate(response="x" * 100)
    assert result.score == 1.0
    assert result.passed is True


async def test_response_at_double_limit_scores_zero(evaluator):
    result = await evaluator.evaluate(response="x" * 200)
    assert result.score == 0.0
    assert result.passed is False


async def test_response_halfway_over_scores_half(evaluator):
    result = await evaluator.evaluate(response="x" * 150)
    assert abs(result.score - 0.5) < 0.001
    assert result.passed is False  # 0.5 < threshold 0.8


async def test_details_include_char_count(evaluator):
    result = await evaluator.evaluate(response="Hello!")
    assert result.details["char_count"] == 6
    assert result.details["max_chars"] == 100
    assert result.details["over_by"] == 0


async def test_over_details_show_overage(evaluator):
    result = await evaluator.evaluate(response="x" * 130)
    assert result.details["over_by"] == 30


async def test_returns_validation_result(evaluator):
    result = await evaluator.evaluate(response="any response")
    assert isinstance(result, ValidationResult)


async def test_audit_log_called_once(evaluator, mock_audit):
    await evaluator.evaluate(response="test")
    mock_audit.log.assert_called_once()


async def test_audit_id_matches_result(evaluator, mock_audit):
    result = await evaluator.evaluate(response="test")
    logged_entry = mock_audit.log.call_args[0][0]
    assert logged_entry.audit_id == result.audit_id


def test_evaluator_name(evaluator):
    assert evaluator.name == "conciseness_evaluator"
```

Run the tests:

```bash
poetry run pytest tests/unit/test_conciseness_evaluator.py -v
```

### Step 4: Use It With TestRunner

```python
import asyncio
from agentproof.core.config import AgentProofConfig
from agentproof.core.audit import AuditLogger
from agentproof.core.runner import TestRunner
from agentproof.evaluators.conciseness import ConcisenessEvaluator


async def main() -> None:
    config = AgentProofConfig()
    audit_logger = AuditLogger(log_dir=config.audit_log_dir)

    evaluator = ConcisenessEvaluator(config, audit_logger, max_chars=300, threshold=0.8)

    runner = TestRunner(config)
    runner.register(evaluator, response="This is the model's response to the user query.")

    summary = await runner.run()

    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.metric}: score={result.score:.2f} "
              f"threshold={result.threshold} latency={result.latency_ms:.0f}ms")
        if result.details:
            print(f"       {result.details}")
        if result.error:
            print(f"       ERROR: {result.error}")

    print(f"\nPass rate: {summary.pass_rate:.1%} | All passed: {summary.all_passed}")


asyncio.run(main())
```

Output:

```
[PASS] conciseness: score=1.00 threshold=0.8 latency=0ms
       {'char_count': 47, 'max_chars': 300, 'over_by': 0}

Pass rate: 100.0% | All passed: True
```

The audit log entry for this run is written to `./audit_logs/audit_YYYY-MM-DD.jsonl` automatically.

---

## 8. What Comes Next: Phase 2 Preview

Phase 2 implements `src/agentproof/integrations/anthropic.py` and `src/agentproof/evaluators/llm.py`. These are the first files that make AgentProof a working AI evaluation framework — not just infrastructure.

> **Status:** Phase 2 is complete and tested (157 unit tests). Live scoring still requires `ANTHROPIC_API_KEY` (or another judge key) because DeepEval metrics call a judge model.

### What Phase 2 Adds

**`AnthropicIntegration`** (`src/agentproof/integrations/anthropic.py`) — a thin async wrapper around the Anthropic SDK. It initializes `AsyncAnthropic` from config, constructs messages, calls `client.messages.create(...)`, and returns parsed content. This isolates all Anthropic SDK usage to one file, making the `LLMEvaluator` easy to test with mocks.

**`LLMEvaluator`** (`src/agentproof/evaluators/llm.py`) — a `BaseEvaluator` subclass with three evaluation methods, each backed by a DeepEval metric:

| Method | DeepEval Metric | Threshold | What it measures |
|---|---|---|---|
| `evaluate_relevance(input, output)` | `AnswerRelevancyMetric` | 0.7 | Does the output answer the question? |
| `evaluate_faithfulness(input, output, context)` | `FaithfulnessMetric` | 0.9 | Is the output grounded in provided documents? |
| `evaluate_hallucination(input, output, context)` | `HallucinationMetric` | 0.1 | Hallucination *rate* (lower is better). DeepEval 4.x returns alignment; AgentProof stores `1.0 - score`. |
| `evaluate_toxicity(input, output)` | `ToxicityMetric` | 0.1 | Toxicity *rate* (lower is better). DeepEval already returns a rate; AgentProof does not invert it. |

**The DeepEval pattern Phase 2 uses:**

```python
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.runner import TestRunner
from agentproof.evaluators.llm import LLMEvaluator

config = AgentProofConfig()
runner = TestRunner(config)
evaluator = LLMEvaluator(config, runner.audit_logger)

runner.register(
    evaluator,
    input="What is our refund policy?",
    output="Refunds are available within 30 days of purchase.",
    metric="relevance",
)
runner.register(
    evaluator,
    input="What is our refund policy?",
    output="Refunds are available within 30 days of purchase.",
    context=["Our policy allows refunds within 30 days of purchase date."],
    metric="faithfulness",
)
runner.register(
    evaluator,
    input="What is our refund policy?",
    output="Refunds are available within 30 days of purchase.",
    context=["Our policy allows refunds within 30 days of purchase date."],
    metric="hallucination",
)

summary = await runner.run()
# summary.results → ValidationResult objects
# summary.all_passed → CI exit signal
```

`evaluate_all(...)` runs the four metrics (relevance, faithfulness, hallucination, toxicity) without registering four times. If `output` is omitted and an `AnthropicIntegration` is passed into `LLMEvaluator`, the response is generated once and reused.

### Why Phase 2 Is the First Demonstrable Milestone

Phase 2 is complete. You can demonstrate:

1. Pass a user query, a model response, and source documents to `LLMEvaluator`
2. Four DeepEval metrics run against the response using Claude as judge
3. Four scored `ValidationResult` objects come back, each with a `passed` flag
4. Four tamper-evident `AuditEntry` records are written to the daily JSONL log (PII redacted)
5. A `TestRunSummary` is returned with a pass rate and an `all_passed` flag for CI integration. Prometheus counters and an `EvalTrace` are recorded on the runner.

This is the point at which AgentProof produces visible, auditable output from real AI evaluation — not just infrastructure scaffolding.

Phase 5 is complete: `GovernanceValidator` checks checkpoint coverage, human-override record/apply, structural separation, and SHA-256 tamper evidence. `StreamingValidator` checks TTFT (< 2s), throughput (> 20 tok/s), stream completeness, graceful degradation, and mid-stream error handling.

Phase 6 is complete: `PostgresValidator` checks schema integrity, agent-state persistence, transaction rollback, write latency, and silent writes. `RedisValidator` checks semantic cache correctness, TTL, invalidation, and session state. `DataLayerValidator` compares Redis cache values to the PostgreSQL source of truth.

Evals, guardrails, and observability (complete, tested): `EvalHarness` runs suites of `EvalCase`s. `GuardrailValidator` blocks PII, prompt injection, policy phrases, and off-allowlist tools. `WorkflowEvaluator` scores a *recorded* agentic trace (subagents, skills, MCP tools, background jobs, PR-review gates) — AgentProof does not host those runtimes. `MetricsRegistry` exports Prometheus text; `TraceStore` keeps in-memory spans. Audit JSONL redacts PII before hashing.

Phase 7 is complete: `GraphValidator` checks entity resolution (sources collapse to one node), relationship type and direction, temporal event order, known Cypher result ids, and whether injected anomalies (missing / duplicate / contradictory / malformed) are surfaced. `Neo4jIntegration` talks Bolt (Neo4j and Memgraph).

Next: Phase 8 — Nango connector / ingestion validation.

---

*AgentProof — Enterprise-grade AI Agent Testing and Evaluation Framework*
*Ana Bruno — ThinkAstra Consulting, San Diego CA*
*Phases 1–7 complete and tested. Phase 8 not started.*
