# CLAUDE.md — AgentProof
> Read this before touching anything. This file is authoritative.

---

## What This Project Is

AgentProof is an AI systems reliability and evaluation platform built by Ana Bruno (ThinkAstra Consulting). It evaluates non-deterministic AI outputs — LLM responses, RAG pipelines, multi-model routing, knowledge graph integrity, and governance audit trails — in environments where assertion-based checks fail.

**This is both a portfolio project and a potential ThinkAstra product.**
Every component must be accurate, defensible, and demo-ready.

---

## Absolute Constraints — Non-Negotiable

### NEVER add these — hard stops, no exceptions:
- `langchain` — any import, any version, any subpackage
- `langgraph` — any import, any version
- `langsmith` — use Langfuse or custom logging instead
- `langbite` — does not exist as a library
- Raw `dict` return types from evaluators — use typed dataclasses only
- Synchronous LLM calls in async contexts — always `await`
- API keys in code — environment variables only, always

### ALWAYS do these without being asked:
- Update `CHANGELOG.md` after every session
- Update `STATUS.md` after every session
- Write tests for every component you implement
- Add docstrings to every class and method
- Run `poetry run pytest tests/unit/ -v` after significant changes
- Keep `README.md` accurate — never describe unimplemented features as complete

---

## Documentation Protocol

### CHANGELOG.md format:
```
## [YYYY-MM-DD]
### [Component]
- What changed
- Why it changed
- What it affects
```

### STATUS.md format:
```
## Current Status — [DATE]
### Complete and runnable:
### Partially implemented:
### Not started:
### Next priority:
```

### Inline comments:
- Every class: docstring with purpose, usage example
- Every method: docstring with params, return type, what it evaluates
- Every stub: `# TODO: implement [what] using [how] — estimated [time]`

---

## Build Order — Follow This Exactly

```
Phase 1 — Core infrastructure:
  src/agentproof/core/config.py
  src/agentproof/core/errors.py
  src/agentproof/core/retry.py
  src/agentproof/core/audit.py
  src/agentproof/core/base.py
  src/agentproof/core/runner.py
  tests/unit/ for all of the above

Phase 2 — LLM evaluation:
  src/agentproof/integrations/anthropic.py
  src/agentproof/evaluators/llm.py
  tests/unit/test_llm_evaluator.py

Phase 3 — RAG + vector store:
  src/agentproof/integrations/qdrant.py
  src/agentproof/evaluators/rag.py
  tests/unit/test_rag_evaluator.py
  tests/unit/test_qdrant_evaluator.py

Phase 4 — Routing validation:
  src/agentproof/integrations/litellm.py
  src/agentproof/evaluators/routing.py
  tests/unit/test_routing_validator.py

Phase 5 — Governance + streaming:
  src/agentproof/validators/governance.py
  src/agentproof/evaluators/streaming.py
  tests/unit/test_governance_validator.py
  tests/unit/test_streaming_validator.py

Phase 6 — Data layer:
  src/agentproof/integrations/postgres.py
  src/agentproof/integrations/redis.py
  src/agentproof/validators/data_layer.py
  tests/unit/test_postgres_validator.py
  tests/unit/test_redis_validator.py

Alongside Phase 6 (complete, tested — not a skipped numbered phase):
  src/agentproof/core/safety.py
  src/agentproof/core/observability.py
  src/agentproof/validators/guardrails.py
  src/agentproof/evaluators/harness.py
  src/agentproof/evaluators/workflow.py
  tests/unit/test_safety.py
  tests/unit/test_guardrail_validator.py
  tests/unit/test_observability.py
  tests/unit/test_workflow_evaluator.py

Phase 7 — Graph validation:
  src/agentproof/integrations/neo4j.py
  src/agentproof/validators/graph.py
  tests/unit/test_graph_validator.py

Phase 8 — Connector validation:
  src/agentproof/integrations/nango.py
  src/agentproof/validators/ingestion.py
  tests/unit/test_nango_validator.py

Phase 9 — Infrastructure validation:
  src/agentproof/integrations/docker.py
  src/agentproof/integrations/gcp.py
  src/agentproof/validators/infrastructure.py
  tests/unit/test_infrastructure_validator.py
  .github/workflows/agentproof.yml
  docker-compose.yml

Phase 10 — Examples + CLI + demo polish:
  examples/quickstart.py
  examples/peachpilot_demo.py
  src/agentproof/cli.py
  README.md
```

Do not skip phases. Do not build Phase 3 before Phase 2 is tested.

---

## Tech Stack

```toml
# Use these — nothing else without checking first
deepeval          # LLM evaluation metrics — primary eval engine
litellm           # multi-model routing client
anthropic         # Claude API — direct SDK only
openai            # GPT — direct SDK only
qdrant-client     # vector database testing
asyncpg           # PostgreSQL async driver
redis             # Redis async client
neo4j             # Neo4j/Memgraph graph DB driver
fastapi           # service layer if needed
pydantic v2       # all data models and config
pytest            # test runner
pytest-asyncio    # async test support
pytest-xdist      # parallel test execution
httpx             # async HTTP client for streaming tests
python-dotenv     # environment variable loading
typer + rich      # CLI
prometheus-client # metrics export
docker            # Docker SDK for container health validation
firebase-admin    # Firebase validation
google-cloud-run  # GCP Cloud Run validation
```

---

## DeepEval Integration Patterns

Always use DeepEval's async methods:

```python
# Correct pattern
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="user query",
    actual_output="model response",
    retrieval_context=["doc1", "doc2"]  # for RAG tests
)

metric = AnswerRelevancyMetric(
    threshold=config.default_relevance_threshold,
    model=config.judge_model
)

await metric.a_measure(test_case)  # always async

result = ValidationResult(
    passed=metric.score >= metric.threshold,
    score=metric.score,
    ...
)
```

### Metric priority order (implement in this order):
1. `AnswerRelevancyMetric`
2. `FaithfulnessMetric`
3. `HallucinationMetric`
4. `ContextualRecallMetric`
5. `ContextualPrecisionMetric`
6. `ToxicityMetric`

### Default thresholds (all configurable):
- Relevance: > 0.7
- Faithfulness: > 0.9 (executive-facing: > 0.95)
- Hallucination rate: < 0.1
- Contextual recall: > 0.7
- Contextual precision: > 0.8
- Toxicity rate: < 0.1

---

## ValidationResult — Always Use This

```python
@dataclass
class ValidationResult:
    passed: bool
    score: float                    # 0.0 – 1.0
    evaluator_name: str
    metric: str
    threshold: float
    details: Dict[str, Any]
    latency_ms: float
    timestamp: datetime
    error: Optional[str]            # set on failure — never raise
    audit_id: str                   # always linked to audit log entry
```

Never return a raw dict from an evaluator. Always return ValidationResult.

---

## Audit Logging — Mandatory

Every evaluation result must produce an audit entry. No exceptions.

```python
# AuditLogger writes this to audit_logs/audit_YYYY-MM-DD.jsonl
{
  "audit_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "evaluator": "LLMEvaluator",
  "metric": "faithfulness",
  "score": 0.94,
  "threshold": 0.90,
  "passed": true,
  "model": "claude-sonnet-4-5",
  "latency_ms": 342.1,
  "entry_hash": "sha256:..."     # tamper evidence
}
```

AuditError is the only error that propagates — never swallow audit failures.

---

## Mock Strategy for Unit Tests

All external dependencies must be mocked in unit tests:

```python
# Always mock these in unit tests
@pytest.fixture
def mock_anthropic():
    with patch("agentproof.integrations.anthropic.AsyncAnthropic") as m:
        yield m

@pytest.fixture
def mock_qdrant():
    with patch("agentproof.integrations.qdrant.QdrantClient") as m:
        yield m

@pytest.fixture
def mock_litellm():
    with patch("agentproof.integrations.litellm.litellm.acompletion") as m:
        yield m

@pytest.fixture
def mock_deepeval_metric():
    with patch("deepeval.metrics.AnswerRelevancyMetric") as m:
        m.return_value.score = 0.85
        m.return_value.reason = "mocked reason"
        yield m
```

---

## Interview Accuracy Standard

This project is used in technical interviews. The following rules apply:

- Only describe components as complete if they have passing tests
- README must reflect actual implementation status — use the status table
- Demo means: runs end to end, produces scored output, no stubs hit
- If asked "can I see it run?" — it must run
- CHANGELOG and STATUS must be current before any demo

---

## Project Context

**Owner:** Ana Bruno — ThinkAstra Consulting  
**Purpose:** Portfolio project + potential ThinkAstra product  
**Primary target:** Enterprise AI systems reliability — insurance, finance, healthcare  
**Competitive positioning:** Evaluates what assertion-based tooling cannot — non-deterministic agent outputs, multi-model routing, knowledge graph integrity, governance audit trails  
**GitHub:** github.com/anabelenp  
**Contact:** ThinkAstra Consulting, San Diego CA
