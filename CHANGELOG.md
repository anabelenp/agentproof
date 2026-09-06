## [2026-09-06]
### Phase 6 — Data layer
- Implemented `PostgresIntegration` + `PostgresValidator` (`src/agentproof/integrations/postgres.py`) — asyncpg pool wrapper; schema integrity (tables/columns/types/constraints), agent-state persistence, transaction atomicity with injected failure + rollback probe, write latency vs `max_db_write_latency_ms` (default 200ms), silent-write detection (status-ok but row missing)
- Implemented `RedisIntegration` + `RedisValidator` (`src/agentproof/integrations/redis.py`) — redis.asyncio wrapper; cache correctness via semantic equality (JSON/numeric/whitespace), TTL within `ttl_tolerance_seconds` (default 5s), opt-in expiry confirmation, cache invalidation after upstream update, session state via hash or JSON blob
- Implemented `DataLayerValidator` (`src/agentproof/validators/data_layer.py`) — dispatches to postgres/redis metrics and scores cache-vs-PostgreSQL source consistency
- Added `postgres_url`, `postgres_pool_size`, `redis_url`, `redis_ttl_seconds`, `max_db_write_latency_ms`, `ttl_tolerance_seconds` to `AgentProofConfig`
- Added `DataLayerValidatorError`; added `asyncpg` and `redis` dependencies
- Added unit tests: `tests/unit/test_postgres_validator.py`, `tests/unit/test_redis_validator.py` (436 unit tests total, all mocked)

### Phase 5 — Governance + streaming
- Implemented `GovernanceValidator` (`src/agentproof/validators/governance.py`) — audit trail completeness (expected checkpoints + optional SHA-256), human-override record/apply, structural separation, tamper evidence
- Implemented `StreamingValidator` (`src/agentproof/evaluators/streaming.py`) — TTFT vs `max_ttft_seconds`, throughput vs `min_token_throughput`, completeness (done marker), graceful degradation (visible failure vs silent empty 200), mid-stream error handling
- Streaming accepts pre-recorded `chunks` for unit tests or live POST via httpx (SSE `data:` lines)
- Extended `AuditLogger` with `read_entries()` and `verify_entry_data()` so governance can load and hash-check JSONL trails
- Added unit tests: `tests/unit/test_governance_validator.py`, `tests/unit/test_streaming_validator.py` (324 unit tests total, all mocked)

### Phase 4 — Routing validation
- Implemented `LiteLLMIntegration` (`src/agentproof/integrations/litellm.py`) — async wrapper around `litellm.acompletion`, text/model extraction, retry on transient errors, single-attempt primary then fallback failover
- Implemented `RoutingValidator` (`src/agentproof/evaluators/routing.py`) — routing correctness (silent misroute detection), fallback within `max_fallback_seconds` (default 5s), output consistency via `LLMEvaluator` with `routing_consistency_delta` (default 0.15), cost-tier routing
- Model matching ignores provider prefixes (`anthropic/claude-...`) and does not conflate `gpt-4` with `gpt-4o`
- Added `max_fallback_seconds` and `routing_consistency_delta` to `AgentProofConfig`
- Added unit tests: `tests/unit/test_routing_validator.py` (269 unit tests total, all mocked)

### Phase 3 — RAG + vector store
- Implemented `QdrantIntegration` (`src/agentproof/integrations/qdrant.py`) — async wrapper around `QdrantClient.query_points` / `count` / `retrieve` via `asyncio.to_thread`, retry on transient errors
- Implemented `QdrantEvaluator` — id-level precision@k / recall@k (`retrieval_quality`), collection integrity, retrieval latency vs `max_retrieval_latency_ms` (default 500ms)
- Implemented `RAGEvaluator` (`src/agentproof/evaluators/rag.py`) — DeepEval `ContextualRecallMetric` + `ContextualPrecisionMetric`; generation faithfulness/relevance delegated to `LLMEvaluator`
- Empty retrieval is a quality failure (score 0, `error` unset), not an evaluator error
- `evaluate_full_pipeline` optionally searches Qdrant, scores id-level retrieval, then runs the four RAG metrics
- Added `max_retrieval_latency_ms` to `AgentProofConfig`
- Added unit tests: `tests/unit/test_qdrant_evaluator.py`, `tests/unit/test_rag_evaluator.py` (222 unit tests total, all mocked)

### Phase 2 — LLM evaluation
- Implemented `AnthropicIntegration` (`src/agentproof/integrations/anthropic.py`) — async wrapper around `AsyncAnthropic.messages.create`, text extraction, retry on transient SDK errors, `ConfigurationError` when `ANTHROPIC_API_KEY` is missing
- Implemented `LLMEvaluator` (`src/agentproof/evaluators/llm.py`) — DeepEval `AnswerRelevancyMetric`, `FaithfulnessMetric`, and `HallucinationMetric` behind `ValidationResult` + audit logging
- Hallucination is reported as a rate (lower is better, default threshold 0.1). DeepEval 4.x's higher-is-better alignment score is inverted: `rate = 1.0 - deepeval_score`
- `evaluate()` dispatches by metric name; `evaluate_all()` runs the three Phase 2 metrics; optional `AnthropicIntegration` generates `output` when the caller omits it
- Claude judges are wrapped in DeepEval `AnthropicModel` when an API key is present
- Evaluation failures become error results; `AuditError` still propagates
- Added unit tests: `tests/unit/test_anthropic.py`, `tests/unit/test_llm_evaluator.py` (157 unit tests total, all mocked)

### Tests
- Isolated `test_default_api_keys_are_none` from process-level `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` so defaults are hermetic

## [2026-05-25]
### Documentation
- Added `TUTORIAL.md` — comprehensive onboarding guide for QA engineers new to AI evaluation
  - Covers all 17 tech stack components with plain-English explanations and traditional QA analogies
  - Explains non-determinism, behavioral contracts, LLM-as-a-Judge, RAG, prompt regression, multi-model routing
  - Full architecture walkthrough with three ASCII diagrams: data flow, evaluator hierarchy, CI/CD trigger matrix
  - Deep dive into all six Phase 1 components with complete field references and usage examples
  - Step-by-step custom evaluator tutorial with complete, runnable code and full test suite
  - Phase 2 preview — what `LLMEvaluator` will add and why it is the first demonstrable milestone
  - AI TESTING DIFFERENCE callout blocks wherever AI evaluation fundamentally differs from traditional QA
  - Accuracy-constrained: only documents Phase 1 as complete; all other phases clearly marked not yet implemented

### Phase 1 — Core Infrastructure
- Scaffolded pyproject.toml with all v1.0 Poetry dependencies and dev tooling
- Implemented `AgentProofConfig` (Pydantic BaseSettings) — thresholds, SLA, retry, API keys via env
- Implemented full error hierarchy: `AgentProofError` → `EvaluatorError`, `AuditError`, `RetryExhaustedError`, `ConfigurationError`, `IntegrationError`, and five evaluator-specific subclasses
- Implemented `RetryConfig` and `retry_async` decorator — exponential backoff with jitter, configurable exception types
- Implemented `AuditLogger` — daily JSONL rotation, SHA-256 tamper evidence per entry, asyncio.Lock for concurrent writes, integrity verification
- Implemented `ValidationResult` dataclass — score validated [0.0, 1.0], audit_id linked to log entry
- Implemented `BaseEvaluator` abstract class — `_write_audit`, `_new_audit_id`, `_error_result`, `_elapsed_ms` helpers
- Implemented `TestRunner` + `TestRunSummary` — parallel/sequential modes, one failing evaluator never aborts pipeline, pass_rate and all_passed properties
- Added 60+ unit tests across `test_config`, `test_errors`, `test_retry`, `test_audit`, `test_base`, `test_runner`
- Added `conftest.py` with shared `config` and `audit_logger` fixtures
- Added `.env.example`, `.gitignore`, `README.md`, `STATUS.md`
