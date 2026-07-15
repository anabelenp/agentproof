# AgentProof — Design Document
> Version 1.0 | May 2026 | ThinkAstra Consulting

---

## Design Philosophy

### 1. Behavioral contracts over string assertions
LLM outputs cannot be tested with `assert output == expected`. AgentProof tests *behavioral contracts*: does the output stay within defined topic boundaries? Does it meet minimum quality thresholds? Does it convey semantically correct meaning regardless of phrasing? This is the foundational shift from traditional QA thinking.

### 2. Fail gracefully, always
One failing evaluator must never abort the pipeline. Enterprise AI systems have many interdependent components — a single point of failure in testing is unacceptable. Every evaluator catches its own errors, logs them, and returns a structured failure result. The pipeline continues.

### 3. Audit everything
Every evaluation run produces an immutable, tamper-evident audit record. This is non-negotiable for regulated industries (insurance, finance, healthcare, legal). No evaluation result exists without a corresponding audit entry.

### 4. Pluggable by design
Model providers, vector stores, graph databases, and connector layers are all pluggable. AgentProof tests the behavior of any system in these categories, not a specific vendor's implementation. This is how it stays relevant as the AI stack evolves.

### 5. No LangChain, ever
LangChain and LangGraph introduce abstractions that obscure the systems being tested and create defensive interview situations. AgentProof calls AI providers directly — Anthropic SDK, OpenAI SDK, LiteLLM — and owns every layer of its implementation.

---

## Core Design Patterns

### Pattern 1: BaseEvaluator + ValidationResult

Every evaluator in AgentProof inherits from `BaseEvaluator` and returns a `ValidationResult`. This enforces a consistent async interface across all evaluation types — deterministic, non-deterministic, and hybrid.

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
    error: Optional[str]            # set on failure, never raised
    audit_id: str                   # links to AuditLogger entry
```

### Pattern 2: Threshold-based scoring

Every evaluator defines a threshold. Pass/fail is determined by whether the score meets that threshold. Thresholds are configurable per use case — an executive-facing finding requires faithfulness > 0.95; an internal draft requires > 0.70.

```python
class EvaluatorConfig(BaseModel):
    threshold: float = 0.7
    model: str = "gpt-4o"
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
```

### Pattern 3: LLM-as-a-Judge

For non-deterministic outputs, AgentProof uses a separate LLM to score the output of the system under test. The judge model is configurable and independent of the model being evaluated. DeepEval provides the metric implementations; AgentProof provides the harness and enterprise integration layer.

### Pattern 4: Prompt regression suite

Every prompt change or model update must run the full regression suite before deployment. AgentProof maintains a library of canonical test cases — defined inputs with behavioral expectations. These run automatically in CI/CD.

```python
@dataclass
class RegressionTestCase:
    id: str
    input: str
    expected_behavior: str          # natural language behavioral contract
    retrieval_context: List[str]    # for RAG tests
    min_relevance: float
    min_faithfulness: float
    max_hallucination: float
    tags: List[str]
```

### Pattern 5: Unified pipeline with pluggable evaluators

The `TestRunner` orchestrates evaluation across all evaluator types in a single execution. Each evaluator is registered, configured, and run in parallel where possible. Results are aggregated into a single report and a single audit record.

---

## Key Component Designs

### AgentProofConfig

Single Pydantic settings object that controls all framework behavior. Loaded from environment variables or a YAML file. Never hardcoded.

```python
class AgentProofConfig(BaseSettings):
    # Model providers
    anthropic_api_key: SecretStr
    openai_api_key: SecretStr

    # Vector store — Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[SecretStr] = None

    # Graph database — Neo4j / Memgraph (Bolt protocol compatible)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    # Relational database — PostgreSQL
    postgres_url: str = "postgresql://localhost:5432/agentproof_test"
    postgres_pool_size: int = 5

    # Cache — Redis
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600

    # Connector layer — Nango
    nango_api_key: Optional[SecretStr] = None
    nango_base_url: str = "https://api.nango.dev"

    # Infrastructure — GCP
    gcp_project_id: Optional[str] = None
    gcp_region: str = "us-central1"
    cloud_run_service_url: Optional[str] = None
    firebase_credentials_path: Optional[Path] = None

    # Thresholds (all configurable)
    default_relevance_threshold: float = 0.7
    default_faithfulness_threshold: float = 0.9
    default_hallucination_threshold: float = 0.1
    executive_faithfulness_threshold: float = 0.95

    # Performance SLAs
    max_ttft_seconds: float = 2.0
    min_token_throughput: float = 20.0
    max_retrieval_latency_ms: float = 500.0
    max_cloud_run_cold_start_seconds: float = 10.0
    max_db_write_latency_ms: float = 200.0

    # Audit
    audit_log_dir: Path = Path("audit_logs")
    audit_hash_algorithm: str = "sha256"

    model_config = SettingsConfigDict(env_file=".env")
```

### BaseEvaluator

Abstract base class all evaluators inherit from. Enforces async interface, audit logging, and retry logic.

```python
class BaseEvaluator(ABC):
    def __init__(self, config: AgentProofConfig):
        self.config = config
        self._audit = AuditLogger(config.audit_log_dir)
        self._retry = RetryConfig(max_attempts=3)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def evaluate(self, **kwargs) -> ValidationResult: ...

    async def _run_with_audit(self, **kwargs) -> ValidationResult:
        start = time.monotonic()
        try:
            result = await self.evaluate(**kwargs)
        except Exception as e:
            result = ValidationResult(
                passed=False,
                score=0.0,
                evaluator_name=self.name,
                error=str(e),
                ...
            )
        finally:
            result.latency_ms = (time.monotonic() - start) * 1000
            await self._audit.log(result)
        return result
```

### RoutingValidator

Tests LiteLLM multi-model routing behavior. Three test types:

1. **Routing correctness** — given input characteristics, was the correct model selected?
2. **Fallback behavior** — when primary model fails, does secondary activate within SLA?
3. **Output consistency** — does the same prompt produce semantically equivalent output across models?

```python
class RoutingValidator(BaseEvaluator):
    def __init__(self, config, litellm_config: LiteLLMConfig):
        super().__init__(config)
        self._router = litellm.Router(model_list=litellm_config.models)

    async def validate_routing_correctness(
        self, prompt: str, expected_model: str
    ) -> ValidationResult:
        # Send prompt, inspect which model handled it
        # Compare against expected_model
        ...

    async def validate_fallback(
        self, prompt: str, primary_model: str, fallback_model: str
    ) -> ValidationResult:
        # Mock primary model failure
        # Confirm fallback activates within max_ttft_seconds
        ...

    async def validate_output_consistency(
        self, prompt: str, models: List[str], metric: str = "relevance"
    ) -> ValidationResult:
        # Run same prompt across all models
        # Score each output with DeepEval
        # Confirm all scores meet threshold
        # Report max delta between models
        ...
```

### LLMEvaluator

DeepEval integration layer. Wraps DeepEval metrics with AgentProof's config, audit, and retry infrastructure.

```python
class LLMEvaluator(BaseEvaluator):
    async def evaluate_relevance(
        self, input: str, output: str
    ) -> ValidationResult:
        test_case = LLMTestCase(input=input, actual_output=output)
        metric = AnswerRelevancyMetric(
            threshold=self.config.default_relevance_threshold,
            model=self.config.judge_model
        )
        await metric.a_measure(test_case)
        return ValidationResult(
            passed=metric.score >= metric.threshold,
            score=metric.score,
            evaluator_name=self.name,
            metric="answer_relevancy",
            threshold=metric.threshold,
            details={"reason": metric.reason},
            ...
        )

    async def evaluate_faithfulness(self, ...) -> ValidationResult: ...
    async def evaluate_hallucination(self, ...) -> ValidationResult: ...
    async def evaluate_toxicity(self, ...) -> ValidationResult: ...
```

### RAGEvaluator

Tests retrieval quality and generation faithfulness for RAG pipelines.

```python
class RAGEvaluator(BaseEvaluator):
    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        ground_truth_docs: List[str]
    ) -> ValidationResult:
        # DeepEval ContextualRecallMetric + ContextualPrecisionMetric
        ...

    async def evaluate_generation(
        self,
        query: str,
        response: str,
        retrieval_context: List[str]
    ) -> ValidationResult:
        # DeepEval FaithfulnessMetric + AnswerRelevancyMetric
        ...

    async def evaluate_full_pipeline(
        self, query: str, rag_system: Any, ground_truth: dict
    ) -> List[ValidationResult]:
        # Runs retrieval + generation evaluation end to end
        ...
```

### QdrantEvaluator

Tests vector database retrieval quality and data integrity.

```python
class QdrantEvaluator(BaseEvaluator):
    def __init__(self, config, qdrant_client: QdrantClient):
        super().__init__(config)
        self._client = qdrant_client

    async def evaluate_retrieval_quality(
        self,
        query: str,
        collection: str,
        ground_truth_ids: List[str],
        top_k: int = 5
    ) -> ValidationResult:
        # Search Qdrant, compare results against ground truth
        # Score: precision@k and recall@k
        ...

    async def evaluate_data_integrity(
        self,
        collection: str,
        expected_count: int,
        sample_ids: List[str]
    ) -> ValidationResult:
        # Confirm record count, spot-check specific IDs
        ...

    async def evaluate_retrieval_latency(
        self,
        query: str,
        collection: str
    ) -> ValidationResult:
        # Measure retrieval time, compare against SLA
        ...
```

### GovernanceValidator

Tests human-in-the-loop audit trail integrity and override behavior.

```python
class GovernanceValidator(BaseEvaluator):
    async def validate_audit_trail_completeness(
        self,
        agent_run_id: str,
        expected_checkpoints: List[str]
    ) -> ValidationResult:
        # Load audit log for run_id
        # Confirm all expected checkpoints are present
        # Verify SHA-256 hash integrity on each entry
        ...

    async def validate_human_override(
        self,
        agent_run_id: str,
        override_event: OverrideEvent
    ) -> ValidationResult:
        # Confirm override was logged correctly
        # Confirm downstream agent behavior changed after override
        ...

    async def validate_structural_separation(
        self,
        agent_a: str,
        agent_b: str,
        forbidden_actions: List[str]
    ) -> ValidationResult:
        # Confirm agent_a cannot perform forbidden_actions
        # That are reserved for agent_b
        ...
```

### StreamingValidator

Tests async streaming response behavior and performance.

```python
class StreamingValidator(BaseEvaluator):
    async def validate_ttft(
        self,
        endpoint: str,
        payload: dict,
        max_ttft_seconds: float = None
    ) -> ValidationResult:
        # Time from request send to first token received
        ...

    async def validate_throughput(
        self,
        endpoint: str,
        payload: dict
    ) -> ValidationResult:
        # Tokens per second across full stream
        ...

    async def validate_graceful_degradation(
        self,
        endpoint: str,
        payload: dict,
        mock_failure: bool = True
    ) -> ValidationResult:
        # Simulate model unavailability
        # Confirm graceful error state, not silent failure
        ...
```

### PostgresValidator

Tests PostgreSQL schema integrity, agent state persistence, and transaction atomicity.

```python
class PostgresValidator(BaseEvaluator):
    def __init__(self, config: AgentProofConfig):
        super().__init__(config)
        self._pool: Optional[asyncpg.Pool] = None

    async def validate_schema_integrity(
        self,
        expected_tables: List[str],
        expected_columns: Dict[str, List[str]]
    ) -> ValidationResult:
        # Connect to PostgreSQL
        # Confirm all expected tables exist
        # Confirm column names and types match expected definitions
        ...

    async def validate_agent_state_persistence(
        self,
        agent_run_id: str,
        expected_state_keys: List[str]
    ) -> ValidationResult:
        # Confirm agent state was written correctly
        # Retrieve state for agent_run_id
        # Confirm all expected_state_keys are present and non-null
        ...

    async def validate_transaction_integrity(
        self,
        multi_step_operation: Callable,
        injected_failure_step: int
    ) -> ValidationResult:
        # Execute multi_step_operation
        # Inject failure at injected_failure_step
        # Confirm full rollback — no partial writes committed
        ...

    async def validate_write_latency(
        self,
        table: str,
        payload: Dict
    ) -> ValidationResult:
        # Measure write latency
        # Compare against max_db_write_latency_ms SLA
        ...
```

---

### RedisValidator

Tests Redis cache correctness, TTL behavior, and invalidation patterns used in agent memory and session state.

```python
class RedisValidator(BaseEvaluator):
    def __init__(self, config: AgentProofConfig):
        super().__init__(config)
        self._client: Optional[redis.asyncio.Redis] = None

    async def validate_cache_correctness(
        self,
        key: str,
        expected_value: Any
    ) -> ValidationResult:
        # Retrieve value from Redis
        # Compare against expected_value (semantic equality, not string match)
        ...

    async def validate_ttl_behavior(
        self,
        key: str,
        expected_ttl_seconds: int,
        tolerance_seconds: int = 5
    ) -> ValidationResult:
        # Confirm key TTL matches expected within tolerance
        # Wait for expiry, confirm key is gone
        ...

    async def validate_cache_invalidation(
        self,
        key: str,
        upstream_update: Callable
    ) -> ValidationResult:
        # Confirm value in cache before update
        # Trigger upstream_update (simulates data change)
        # Confirm cache is invalidated — stale value gone
        ...

    async def validate_session_state(
        self,
        session_id: str,
        expected_state: Dict
    ) -> ValidationResult:
        # Retrieve agent session state from Redis
        # Compare against expected_state
        ...
```

---

### GraphValidator

Tests Neo4j/Memgraph knowledge graph integrity — entity resolution, relationship accuracy, and temporal sequencing. Memgraph uses the Bolt protocol and is fully compatible with the Neo4j Python driver.

```python
class GraphValidator(BaseEvaluator):
    def __init__(self, config: AgentProofConfig):
        super().__init__(config)
        self._driver = AsyncGraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_user, config.neo4j_password.get_secret_value())
        )

    async def validate_entity_resolution(
        self,
        entity_sources: List[Dict],   # same entity from CRM, email, calendar
        expected_node_count: int = 1
    ) -> ValidationResult:
        # Confirm multiple source records resolve to one graph node
        # Score: resolved_count / total_expected_entities
        ...

    async def validate_relationship_integrity(
        self,
        cypher_query: str,
        expected_relationship_type: str,
        expected_direction: str
    ) -> ValidationResult:
        # Run Cypher query
        # Confirm relationships match expected type and direction
        ...

    async def validate_temporal_accuracy(
        self,
        entity_id: str,
        expected_event_sequence: List[str]
    ) -> ValidationResult:
        # Retrieve event timeline for entity_id
        # Confirm events are ordered correctly in time
        # Detect out-of-sequence entries
        ...

    async def validate_query_correctness(
        self,
        cypher_query: str,
        expected_node_ids: List[str]
    ) -> ValidationResult:
        # Run Cypher query against known dataset
        # Confirm returned node IDs match expected_node_ids
        ...

    async def validate_failure_modes(
        self,
        collection: str,
        injected_anomalies: List[Dict]
    ) -> ValidationResult:
        # Inject: missing entities, duplicate nodes, contradictory relationships
        # Confirm graph layer surfaces anomalies correctly
        ...
```

---

### NangoValidator

Tests Nango connector reliability, data completeness, silent failure detection, and audit trail continuity across the 700+ connector integration layer.

```python
class NangoValidator(BaseEvaluator):
    def __init__(self, config: AgentProofConfig):
        super().__init__(config)
        self._base_url = config.nango_base_url
        self._api_key = config.nango_api_key.get_secret_value()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"}
        )

    async def validate_connector_reliability(
        self,
        connector_id: str,
        expected_record_count: int,
        sample_fields: List[str]
    ) -> ValidationResult:
        # Trigger sync for connector_id
        # Confirm expected_record_count records ingested
        # Spot-check sample_fields are present and non-null
        ...

    async def validate_silent_failure_detection(
        self,
        connector_id: str,
        injected_failure: str   # "drop_records", "malform_payload", "auth_expire"
    ) -> ValidationResult:
        # Inject failure scenario
        # Confirm system surfaces error — does not silently proceed
        # Score: failure_surfaced (1.0) or silent_failure (0.0)
        ...

    async def validate_schema_drift_handling(
        self,
        connector_id: str,
        original_schema: Dict,
        drifted_schema: Dict
    ) -> ValidationResult:
        # Swap source schema from original to drifted
        # Trigger sync
        # Confirm system handles gracefully — alert, not crash
        ...

    async def validate_auth_token_refresh(
        self,
        connector_id: str
    ) -> ValidationResult:
        # Expire OAuth token
        # Trigger sync
        # Confirm token refresh occurs without data loss
        ...

    async def validate_audit_trail_continuity(
        self,
        connector_id: str,
        sync_run_id: str
    ) -> ValidationResult:
        # Confirm every ingested record traceable from:
        # source system → Nango → knowledge graph → audit log
        ...
```

---

### InfrastructureValidator

Tests Docker container health, GCP Cloud Run deployment correctness, Firebase data layer, and cross-cloud consistency.

```python
class InfrastructureValidator(BaseEvaluator):

    async def validate_docker_health(
        self,
        image_name: str,
        expected_ports: List[int],
        health_check_endpoint: str = "/health"
    ) -> ValidationResult:
        # Start container from image_name
        # Confirm expected_ports are exposed and accepting connections
        # Hit health_check_endpoint — confirm 200 response
        # Measure startup time against SLA
        ...

    async def validate_cloud_run_deployment(
        self,
        service_url: str,
        test_payload: Dict
    ) -> ValidationResult:
        # Send test_payload to Cloud Run service
        # Confirm correct response within latency SLA
        # Measure cold start time if applicable
        ...

    async def validate_firebase_readwrite(
        self,
        collection: str,
        test_document: Dict
    ) -> ValidationResult:
        # Write test_document to Firebase collection
        # Read it back immediately
        # Confirm values match (consistency check)
        # Clean up test document
        ...

    async def validate_cross_cloud_consistency(
        self,
        gcp_read_fn: Callable,
        azure_write_fn: Callable,
        test_payload: Dict
    ) -> ValidationResult:
        # Write test_payload via azure_write_fn (Cosmos DB / AI Search)
        # Read it back via gcp_read_fn
        # Confirm values consistent across cloud boundary
        ...

    async def validate_graceful_degradation(
        self,
        service_url: str,
        simulated_failure: str   # "scale_down", "cold_start", "timeout"
    ) -> ValidationResult:
        # Simulate infrastructure failure condition
        # Confirm system degrades gracefully — correct error state
        # Confirm recovery within defined SLA
        ...
```

---

Orchestrates all evaluators in a single execution pipeline.

```python
class TestRunner:
    def __init__(self, config: AgentProofConfig):
        self._config = config
        self._evaluators: List[BaseEvaluator] = []
        self._results: List[ValidationResult] = []

    def register(self, evaluator: BaseEvaluator) -> "TestRunner":
        self._evaluators.append(evaluator)
        return self

    async def run(self, parallel: bool = True) -> TestRunSummary:
        if parallel:
            results = await asyncio.gather(
                *[e._run_with_audit() for e in self._evaluators],
                return_exceptions=True
            )
        else:
            results = [await e._run_with_audit() for e in self._evaluators]

        self._results = [r for r in results if isinstance(r, ValidationResult)]
        return TestRunSummary(results=self._results)

    def report(self) -> str:
        # Rich terminal output
        ...
```

---

## Audit Design

Every `ValidationResult` is linked to an immutable audit entry via `audit_id`. Audit entries are written to JSONL files with SHA-256 hashes.

```jsonl
{
  "audit_id": "uuid-v4",
  "timestamp": "2026-05-25T10:30:00Z",
  "evaluator": "LLMEvaluator",
  "metric": "faithfulness",
  "input_hash": "sha256:...",
  "output_hash": "sha256:...",
  "score": 0.94,
  "threshold": 0.90,
  "passed": true,
  "model": "claude-sonnet-4-5",
  "latency_ms": 342.1,
  "entry_hash": "sha256:..."
}
```

---

## Error Handling Design

```
EvaluatorError (base)
├── ConfigurationError      # missing keys, invalid thresholds
├── ProviderError           # LLM API failures
│   ├── RateLimitError      # 429 — retry with backoff
│   ├── TimeoutError        # request exceeded timeout
│   └── AuthenticationError # invalid API key — do not retry
├── RetrievalError          # vector store failures (Qdrant)
├── GraphError              # knowledge graph failures (Neo4j/Memgraph)
├── DatabaseError           # PostgreSQL write/read failures
├── CacheError              # Redis failures
├── ConnectorError          # Nango connector failures
│   ├── SilentFailureError  # data dropped without surfacing — escalate
│   └── SchemaDriftError    # source schema changed unexpectedly
├── InfrastructureError     # Docker, GCP, Firebase failures
│   ├── ContainerError      # container health check failed
│   ├── CloudRunError       # Cloud Run deployment or latency failure
│   └── CrossCloudError     # GCP ↔ Azure consistency failure
└── AuditError              # audit logging failures — escalate, never swallow
```

`AuditError` and `SilentFailureError` are the only errors that always propagate — they must never be silently ignored.

---

## CI/CD Design

AgentProof ships with a complete GitHub Actions pipeline. Different suites run at different triggers to balance speed with coverage.

```yaml
# .github/workflows/agentproof.yml
name: AgentProof Test Suite

on:
  push:
    branches: [main, develop]       # unit tests on every push
  pull_request:
    branches: [main]                # unit + integration + docker on PRs
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
        env: {POSTGRES_PASSWORD: test}
        ports: ['5432:5432']
      redis:
        image: redis:7-alpine
        ports: ['6379:6379']
      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/testpassword
          NEO4J_PLUGINS: '["apoc"]'
        ports: ['7687:7687']
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
          NEO4J_URL: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword

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
      - name: Notify on regression failure
        if: failure()
        run: echo "Regression failure — model drift or prompt sensitivity detected"
```

### CI/CD test matrix by trigger:

| Trigger | Unit | Integration (Qdrant/PG/Redis/Neo4j) | Docker | Regression |
|---|---|---|---|---|
| Push to develop | ✅ | ❌ | ❌ | ❌ |
| Pull request | ✅ | ✅ | ✅ | ❌ |
| Merge to main | ✅ | ✅ | ✅ | ❌ |
| Scheduled (6hr) | ❌ | ❌ | ❌ | ✅ |

### Local development with Docker Compose:

```yaml
# docker-compose.yml
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

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/localtest
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - '7687:7687'   # Bolt
      - '7474:7474'   # Browser UI

volumes:
  qdrant_data:
```

```bash
# Start all local dependencies
docker compose up -d

# Run unit tests (no external services needed)
poetry run pytest tests/unit/ -v

# Run integration tests against local services
poetry run pytest tests/integration/ -v

# Run full suite
poetry run pytest tests/ -v --ignore=tests/regression/
```
