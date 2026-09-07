# AgentProof — Architecture
> Version 1.0 | May 2026 | ThinkAstra Consulting

---

## Folder Structure

```
agentproof/
├── SCOPE.md
├── REQUIREMENTS.md
├── DESIGN.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── CHANGELOG.md
├── STATUS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── agentproof.yml
│
├── src/
│   └── agentproof/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # AgentProofConfig (Pydantic BaseSettings)
│       │   ├── base.py             # BaseEvaluator, ValidationResult
│       │   ├── runner.py           # TestRunner, TestRunSummary
│       │   ├── audit.py            # AuditLogger (JSONL + SHA-256 + PII redaction)
│       │   ├── safety.py           # PII / prompt-injection detectors
│       │   ├── observability.py    # Prometheus MetricsRegistry + TraceStore
│       │   ├── retry.py            # RetryConfig, retry_async decorator
│       │   └── errors.py           # EvaluatorError hierarchy
│       │
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── llm.py              # LLMEvaluator (DeepEval: relevance, faithfulness, hallucination, toxicity)
│       │   ├── rag.py              # RAGEvaluator (retrieval + generation)
│       │   ├── routing.py          # RoutingValidator (LiteLLM)
│       │   ├── streaming.py        # StreamingValidator (TTFT, throughput)
│       │   ├── harness.py          # EvalHarness + EvalCase suite runner
│       │   ├── workflow.py         # WorkflowEvaluator (recorded agentic traces)
│       │   └── regression.py       # RegressionSuite, RegressionTestCase (not started)
│       │
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── governance.py       # GovernanceValidator (audit trail, override)
│       │   ├── data_layer.py       # DataLayerValidator (Postgres + Redis coordinator)
│       │   ├── guardrails.py       # PII, injection, policy, tool allowlist
│       │   ├── graph.py            # GraphValidator (entity resolution, relationships)
│       │   ├── ingestion.py        # IngestionValidator (Nango + file ingestion)
│       │   └── trust.py            # TrustValidator (executive output clarity) — not started
│       │
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── qdrant.py           # QdrantEvaluator
│       │   ├── postgres.py         # PostgreSQL schema + state validation
│       │   ├── redis.py            # Redis cache correctness + TTL validation
│       │   ├── neo4j.py            # Neo4jIntegration (async Bolt / Memgraph)
│       │   ├── nango.py            # NangoIntegration (connector HTTP API)
│       │   ├── gcp.py              # GCP Cloud Run + Firebase validation
│       │   ├── docker.py           # Docker container health validation
│       │   ├── litellm.py          # LiteLLM routing client wrapper
│       │   └── anthropic.py        # Direct Anthropic SDK wrapper
│       │
│       ├── reporters/
│       │   ├── __init__.py
│       │   ├── terminal.py         # Rich terminal output
│       │   └── json_reporter.py    # JSON output for CI/CD
│       │
│       └── cli.py                  # Typer CLI entrypoint
│
├── tests/
│   ├── conftest.py                 # Shared fixtures, mocks, strategies
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_base.py
│   │   ├── test_audit.py
│   │   ├── test_retry.py
│   │   ├── test_errors.py
│   │   ├── test_llm_evaluator.py
│   │   ├── test_rag_evaluator.py
│   │   ├── test_routing_validator.py
│   │   ├── test_streaming_validator.py
│   │   ├── test_governance_validator.py
│   │   ├── test_qdrant_evaluator.py
│   │   ├── test_postgres_validator.py
│   │   ├── test_redis_validator.py
│   │   ├── test_safety.py
│   │   ├── test_guardrail_validator.py
│   │   ├── test_workflow_evaluator.py
│   │   ├── test_observability.py
│   │   ├── test_graph_validator.py
│   │   └── test_nango_validator.py
│   │
│   ├── integration/
│   │   ├── test_litellm_routing.py         # Requires LiteLLM + API keys
│   │   ├── test_qdrant_retrieval.py        # Requires Qdrant instance
│   │   ├── test_postgres_validation.py     # Requires PostgreSQL
│   │   ├── test_redis_validation.py        # Requires Redis
│   │   ├── test_neo4j_graph.py             # Requires Neo4j/Memgraph
│   │   ├── test_nango_connectors.py        # Requires Nango credentials
│   │   ├── test_gcp_deployment.py          # Requires GCP credentials
│   │   ├── test_docker_health.py           # Requires Docker
│   │   └── test_full_pipeline.py           # End-to-end evaluation run
│   │
│   ├── regression/
│   │   ├── test_cases/
│   │   │   ├── relevance_suite.yaml    # Canonical relevance test cases
│   │   │   ├── faithfulness_suite.yaml # Canonical faithfulness test cases
│   │   │   └── routing_suite.yaml      # Canonical routing test cases
│   │   └── test_regression_runner.py
│   │
│   └── fixtures/
│       ├── sample_documents.py
│       ├── sample_agent_logs.py
│       └── mock_responses.py
│
├── examples/
│   ├── quickstart.py               # 5-minute getting started example
│   ├── peachpilot_demo.py          # Demo against Peach Pilot-style architecture
│   ├── rag_pipeline_eval.py        # RAG evaluation walkthrough
│   └── routing_validation.py       # Multi-model routing validation walkthrough
│
└── docs/
    ├── TUTORIAL.md                 # Developer tutorial (mirrors existing framework)
    ├── EVALUATOR_GUIDE.md          # How to write custom evaluators
    └── ENTERPRISE_GUIDE.md         # Regulated industry deployment guide
```

---

## Component Dependency Map

```
CLI (cli.py)
    │
    └── TestRunner (core/runner.py)
            │
            ├── AgentProofConfig (core/config.py)
            │
            ├── AuditLogger (core/audit.py) ◄─── all evaluators write here
            │
            ├── LLMEvaluator (evaluators/llm.py)
            │       └── DeepEval metrics (incl. toxicity)
            │       └── Anthropic SDK / OpenAI SDK
            │
            ├── RAGEvaluator (evaluators/rag.py)
            │       └── DeepEval metrics
            │       └── QdrantEvaluator (integrations/qdrant.py)
            │
            ├── RoutingValidator (evaluators/routing.py)
            │       └── LiteLLM (integrations/litellm.py)
            │
            ├── StreamingValidator (evaluators/streaming.py)
            │       └── httpx async client
            │
            ├── EvalHarness (evaluators/harness.py)
            │       └── TestRunner + optional GuardrailValidator
            │       └── MetricsRegistry + TraceStore
            │
            ├── WorkflowEvaluator (evaluators/workflow.py)
            │       └── recorded traces (subagents, skills, MCP, background, PR review)
            │
            ├── GovernanceValidator (validators/governance.py)
            │       └── AuditLogger (reads audit logs)
            │
            ├── GuardrailValidator (validators/guardrails.py)
            │       └── core/safety.py (PII + injection detectors)
            │
            ├── DataLayerValidator (validators/data_layer.py)
            │       └── PostgresValidator (integrations/postgres.py)
            │       └── RedisValidator (integrations/redis.py)
            │
            ├── GraphValidator (validators/graph.py)
            │       └── Neo4jIntegration (integrations/neo4j.py)
            │       └── QdrantEvaluator (integrations/qdrant.py)
            │
            ├── IngestionValidator (validators/ingestion.py)
            │       └── NangoIntegration (integrations/nango.py)
            │
            ├── InfrastructureValidator (validators/infrastructure.py)  # not started
            │       └── DockerValidator (integrations/docker.py)
            │       └── GCPValidator (integrations/gcp.py)
            │
            └── TrustValidator (validators/trust.py)              # not started
                    └── LLMEvaluator (reuses LLM judge)

            Observability (every run):
            ├── AuditLogger (core/audit.py) — JSONL + SHA-256 + PII redaction
            └── MetricsRegistry / TraceStore (core/observability.py) — Prometheus text + spans
```

---

## Data Flow

### Standard evaluation run

```
1. User runs: agentproof run --config config.yaml --suite full

2. CLI loads AgentProofConfig from config.yaml + .env

3. TestRunner initializes all registered evaluators

4. For each evaluator (parallel by default):
   a. evaluator.evaluate(**kwargs) called
   b. Result scored against threshold
   c. ValidationResult created with audit_id
   d. AuditLogger writes immutable JSONL entry
   e. Result returned to TestRunner

5. TestRunner aggregates ValidationResults into TestRunSummary

6. Reporter renders terminal output + JSON report

7. Exit code 0 (all passed) or 1 (any failed) for CI/CD
```

### RAG pipeline evaluation flow

```
User query
    │
    ▼
RAGEvaluator.evaluate_full_pipeline()
    │
    ├── QdrantEvaluator.evaluate_retrieval_quality()
    │       │
    │       ├── Search Qdrant collection
    │       ├── Compare results to ground_truth_ids
    │       ├── Score precision@k and recall@k
    │       └── Return ValidationResult + audit entry
    │
    └── LLMEvaluator (via DeepEval)
            │
            ├── FaithfulnessMetric (output grounded in retrieved docs?)
            ├── AnswerRelevancyMetric (output answers the question?)
            └── Return ValidationResult + audit entry
```

### Multi-model routing validation flow

```
Test prompt
    │
    ▼
RoutingValidator.validate_routing_correctness()
    │
    ├── Send prompt via LiteLLM router
    ├── Inspect response headers/metadata for model_used
    ├── Compare model_used against expected_model
    └── Return ValidationResult

RoutingValidator.validate_output_consistency()
    │
    ├── Send same prompt to Model A (Claude)
    ├── Send same prompt to Model B (GPT)
    ├── Score both outputs with LLMEvaluator
    ├── Compare scores — delta must be < threshold
    └── Return ValidationResult
```

---

## Configuration Architecture

```
.env (secrets — never committed)
├── ANTHROPIC_API_KEY
├── OPENAI_API_KEY
├── QDRANT_API_KEY
└── AUDIT_LOG_DIR

config.yaml (non-secret config — committed)
├── thresholds:
│   ├── relevance: 0.7
│   ├── faithfulness: 0.9
│   └── executive_faithfulness: 0.95
├── sla:
│   ├── max_ttft_seconds: 2.0
│   └── min_token_throughput: 20.0
├── qdrant:
│   └── url: http://localhost:6333
└── audit:
    └── log_dir: ./audit_logs

AgentProofConfig (Pydantic BaseSettings)
    └── merges .env + config.yaml + defaults
    └── validates all values on startup
    └── raises ConfigurationError on invalid config
```

---

## Testing Architecture

### Test pyramid

```
                    ┌──────────────┐
                    │  Regression  │  Prompt regression suite
                    │   (slowest)  │  Runs every 6 hours in CI
                    └──────┬───────┘
               ┌───────────┴──────────┐
               │      Integration     │  Requires live services
               │   (Qdrant, LiteLLM)  │  Runs on PR merge
               └───────────┬──────────┘
          ┌─────────────────┴────────────────┐
          │              Unit                │  All mocked, fast
          │  (every component, all mocked)   │  Runs on every commit
          └──────────────────────────────────┘
```

### Mock strategy

All external dependencies are mocked in unit tests:

```python
# conftest.py

@pytest.fixture
def mock_anthropic():
    with patch("agentproof.integrations.anthropic.AsyncAnthropic") as mock:
        mock.return_value.messages.create = AsyncMock(
            return_value=MockAnthropicResponse(content="mocked response")
        )
        yield mock

@pytest.fixture
def mock_qdrant():
    with patch("agentproof.integrations.qdrant.QdrantClient") as mock:
        mock.return_value.search = AsyncMock(
            return_value=[MockQdrantResult(id="doc_1", score=0.92)]
        )
        yield mock

@pytest.fixture
def mock_litellm():
    with patch("agentproof.integrations.litellm.litellm.acompletion") as mock:
        mock.return_value = MockLiteLLMResponse(
            model="claude-sonnet-4-5",
            content="mocked routing response"
        )
        yield mock
```

---

## CLI Architecture

```bash
# Primary commands
agentproof run      # Run evaluation suite
agentproof validate # Validate config file
agentproof init     # Initialize new project
agentproof report   # Generate report from existing results
agentproof version  # Show version

# Examples
agentproof run --config config.yaml --suite full
agentproof run --suite routing --parallel
agentproof run --suite rag --collection my_collection
agentproof run --dry-run
agentproof validate --config config.yaml
agentproof report --format html --input ./audit_logs
```

---

## Extensibility: Custom Evaluators

AgentProof is designed to be extended. Adding a custom evaluator requires:

1. Inherit from `BaseEvaluator`
2. Implement `name` property and `evaluate()` method
3. Return `ValidationResult`
4. Register with `TestRunner`

```python
from agentproof.core.base import BaseEvaluator, ValidationResult

class MyCustomEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "my_custom_evaluator"

    async def evaluate(
        self,
        input: str,
        output: str,
        **kwargs
    ) -> ValidationResult:
        # your evaluation logic here
        score = await self._score(input, output)
        return ValidationResult(
            passed=score >= self.config.default_relevance_threshold,
            score=score,
            evaluator_name=self.name,
            metric="custom_metric",
            threshold=self.config.default_relevance_threshold,
            details={"input_length": len(input)},
            ...
        )

# Register with runner
runner = TestRunner(config)
runner.register(MyCustomEvaluator(config))
await runner.run()
```

---

## CI/CD Architecture

AgentProof ships with a GitHub Actions pipeline that runs different test suites at different triggers:

```yaml
# .github/workflows/agentproof.yml

name: AgentProof Test Suite

on:
  push:
    branches: [main, develop]       # unit tests on every push
  pull_request:
    branches: [main]                # unit + integration on every PR
  schedule:
    - cron: '0 */6 * * *'          # prompt regression every 6 hours

jobs:

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install poetry && poetry install
      - run: poetry run pytest tests/unit/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'pull_request'
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports: ['6333:6333']
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports: ['5432:5432']
      redis:
        image: redis:7
        ports: ['6379:6379']
    steps:
      - uses: actions/checkout@v4
      - run: pip install poetry && poetry install
      - run: poetry run pytest tests/integration/ -v --tb=short
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          QDRANT_URL: http://localhost:6333
          POSTGRES_URL: postgresql://postgres:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379

  docker-validation:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t agentproof:test .
      - run: docker run --rm agentproof:test poetry run pytest tests/unit/ -v

  prompt-regression:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - run: pip install poetry && poetry install
      - run: poetry run pytest tests/regression/ -v --tb=short
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - name: Notify on failure
        if: failure()
        run: echo "Regression failure — model drift or prompt sensitivity detected"
```

### CI/CD test matrix by trigger:

| Trigger | Unit | Integration | Docker | Regression |
|---|---|---|---|---|
| Push to develop | ✅ | ❌ | ❌ | ❌ |
| Pull request | ✅ | ✅ | ✅ | ❌ |
| Merge to main | ✅ | ✅ | ✅ | ❌ |
| Scheduled (6hr) | ❌ | ❌ | ❌ | ✅ |

### Local Docker development:

```bash
# Start local dependencies
docker compose up -d qdrant postgres redis

# Run unit tests
poetry run pytest tests/unit/ -v

# Run integration tests against local services
QDRANT_URL=http://localhost:6333 poetry run pytest tests/integration/ -v

# Build and test container
docker build -t agentproof:local .
docker run --rm agentproof:local poetry run pytest tests/unit/
```

```yaml
# docker-compose.yml — local development
version: '3.9'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ['6333:6333']
    volumes: ['qdrant_data:/qdrant/storage']

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: agentproof_test
      POSTGRES_PASSWORD: localtest
    ports: ['5432:5432']

  redis:
    image: redis:7-alpine
    ports: ['6379:6379']

volumes:
  qdrant_data:
```

Build in this exact sequence — each layer depends on the one before it:

```
Week 1:
  Day 1-2:  core/ — config, base, errors, retry, audit
  Day 3-4:  evaluators/llm.py — DeepEval integration
  Day 5:    tests/unit/ for all core + llm evaluator

Week 2:
  Day 1-2:  evaluators/rag.py + integrations/qdrant.py
  Day 3:    evaluators/routing.py + integrations/litellm.py
  Day 4:    validators/governance.py
  Day 5:    evaluators/streaming.py + core/runner.py

Week 3:
  Day 1-2:  tests/integration/ — live service tests
  Day 3:    examples/ — quickstart + peachpilot_demo
  Day 4:    CLI + reporters
  Day 5:    README + TUTORIAL.md + demo polish

Week 4:
  Day 1-2:  tests/regression/ — canonical test case suite
  Day 3-4:  validators/trust.py + validators/graph.py
  Day 5:    Full end-to-end demo run + STATUS.md update
```

---

## Version Roadmap

| Version | Focus | Timeline |
|---|---|---|
| v1.0 | Core evaluators, LiteLLM, RAG, governance, audit | Weeks 1-4 |
| v1.1 | Memgraph/Neo4j, Nango, bias detection, dashboard | Weeks 5-8 |
| v2.0 | ThinkAstra managed service, API layer, multi-tenant | TBD |
