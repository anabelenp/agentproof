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
