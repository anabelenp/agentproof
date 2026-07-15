# AgentProof — Requirements
> Version 1.0 | May 2026 | ThinkAstra Consulting

---

## Functional Requirements

### FR-01: Multi-Model Routing Validation
**Priority:** Critical
**Covers:** LiteLLM, custom model routers

- FR-01.1: Validate that routing rules direct requests to the correct model given defined input characteristics
- FR-01.2: Validate fallback behavior when primary model is unavailable — secondary model must activate within defined latency threshold
- FR-01.3: Validate output quality consistency across routed models — same prompt routed to Claude vs GPT must meet the same quality bar
- FR-01.4: Validate cost-optimized routing decisions — low-complexity tasks must not be routed to high-cost models
- FR-01.5: Detect silent routing failures — requests that return 200 but were handled by wrong model

**Acceptance criteria:**
- Routing correctness rate ≥ 99% across test suite
- Fallback activation confirmed within 5 seconds
- Output quality delta between models < defined threshold per metric

---

### FR-02: LLM Output Quality Evaluation
**Priority:** Critical
**Covers:** Claude API, OpenAI GPT, any model via LiteLLM

- FR-02.1: Score output relevance to input query (DeepEval AnswerRelevancyMetric)
- FR-02.2: Score output faithfulness to retrieved context (DeepEval FaithfulnessMetric)
- FR-02.3: Detect hallucinations — claims not grounded in source documents (DeepEval HallucinationMetric)
- FR-02.4: Score toxicity of generated content (DeepEval ToxicityMetric)
- FR-02.5: Run prompt regression tests — canonical input/output pairs must pass on every deployment
- FR-02.6: Detect model drift — statistically significant quality degradation across model versions
- FR-02.7: Support LLM-as-a-Judge pattern — use one model to evaluate another's output

**Acceptance criteria:**
- Relevance score threshold configurable per use case (default: > 0.7)
- Faithfulness threshold configurable (default: > 0.9 for executive-facing outputs)
- Hallucination rate < 0.1 in production
- Prompt regression suite runs in CI/CD on every deployment

---

### FR-03: RAG Pipeline Evaluation
**Priority:** Critical
**Covers:** Qdrant, Pinecone, Weaviate, any vector store

- FR-03.1: Evaluate retrieval precision — what fraction of retrieved documents are relevant?
- FR-03.2: Evaluate retrieval recall — what fraction of relevant documents were retrieved?
- FR-03.3: Evaluate contextual relevance — are retrieved documents relevant to the query?
- FR-03.4: Evaluate generation faithfulness — is the answer grounded in retrieved context?
- FR-03.5: Evaluate answer relevance — does the answer address the question?
- FR-03.6: Test retrieval latency under load — p95 latency must meet SLA
- FR-03.7: Test failure modes — empty retrieval, low-confidence retrieval, contradictory context

**Acceptance criteria:**
- Precision@5 > 0.8 on ground truth dataset
- Recall@5 > 0.7 on ground truth dataset
- Faithfulness > 0.9 for high-stakes outputs
- Retrieval p95 latency < 500ms

---

### FR-04: Knowledge Graph Integrity Validation
**Priority:** High
**Covers:** Memgraph, Neo4j, Qdrant (as graph store)

- FR-04.1: Validate entity resolution — same real-world entity appearing in multiple source systems must resolve to one graph node
- FR-04.2: Validate relationship integrity — edges between nodes must be correctly typed and directional
- FR-04.3: Validate temporal accuracy — decision → outcome relationships must be correctly sequenced
- FR-04.4: Validate query correctness — known Cypher/graph queries must return expected subgraphs
- FR-04.5: Test failure modes — missing entities, duplicate nodes, contradictory relationships, malformed data

**Acceptance criteria:**
- Entity resolution accuracy > 95% on synthetic enterprise dataset
- Zero incorrectly typed relationships in test suite
- Temporal sequence accuracy 100% on test cases

---

### FR-05: Governance and Audit Trail Validation
**Priority:** Critical
**Covers:** Multi-agent orchestration layers, human-in-the-loop systems

- FR-05.1: Validate that every agent action produces an audit log entry
- FR-05.2: Validate that human-override checkpoints fire correctly at defined decision points
- FR-05.3: Validate audit log tamper-evidence — SHA-256 hash integrity on every entry
- FR-05.4: Validate governance agent structural separation — agents must not bypass defined oversight boundaries
- FR-05.5: Validate human-override behavior — when a human overrides an agent decision, the system must correctly record and apply the override
- FR-05.6: Test audit trail completeness under failure conditions — partial failures must not produce incomplete audit records

**Acceptance criteria:**
- 100% audit entry coverage — zero unlogged agent actions
- Human-override checkpoint activation confirmed on all defined decision points
- SHA-256 integrity verified on all log entries
- Audit trail complete even when downstream components fail

---

### FR-06: Streaming Response Validation
**Priority:** High
**Covers:** FastAPI async streaming, SSE, WebSocket

- FR-06.1: Validate time-to-first-token (TTFT) meets defined SLA
- FR-06.2: Validate token throughput (tokens/second) meets defined SLA
- FR-06.3: Validate graceful degradation when a model is unavailable or slow
- FR-06.4: Validate streaming response completeness — partial responses must be detected
- FR-06.5: Validate error handling — stream interruptions must produce correct error states, not silent failures

**Acceptance criteria:**
- TTFT < 2 seconds (configurable)
- Token throughput > 20 tokens/second (configurable)
- Graceful degradation confirmed with mock model unavailability
- Zero silent stream failures in test suite

---

### FR-07: Data Ingestion Pipeline Validation
**Priority:** High
**Covers:** Nango connectors, custom ingestion pipelines, file ingestion

- FR-07.1: Validate connector reliability — data flows correctly from each connected system
- FR-07.2: Detect silent ingestion failures — dropped records must surface, not disappear
- FR-07.3: Validate data completeness — after ingestion, all expected records present and correctly mapped
- FR-07.4: Validate schema drift handling — when source system updates field structure, ingestion must handle gracefully
- FR-07.5: Validate file ingestion (Word, Excel, PowerPoint, PDF) including edge cases and encoding issues
- FR-07.6: Validate audit trail continuity through ingestion pipeline

**Acceptance criteria:**
- Data completeness > 99.9% on test dataset
- Silent failure detection rate 100% on injected failures
- Schema drift handling confirmed on 10 defined drift scenarios

---

### FR-08: Trust and Confusion Detection
**Priority:** High
**Covers:** Executive-facing AI outputs, non-technical user outputs

- FR-08.1: Evaluate output clarity — would a non-technical executive understand this finding?
- FR-08.2: Evaluate confidence communication — is uncertainty clearly expressed?
- FR-08.3: Detect contradictory findings — does the output contradict itself?
- FR-08.4: Evaluate actionability — does the output give the user a clear next step?
- FR-08.5: Flag outputs that exceed defined complexity thresholds for executive audiences

**Acceptance criteria:**
- Clarity score > 0.8 on executive output test cases
- Zero self-contradicting outputs in production regression suite
- 100% of high-stakes findings include confidence level

---

### FR-09: Relational and Cache Layer Validation
**Priority:** High
**Covers:** PostgreSQL, Redis

- FR-09.1: Validate PostgreSQL schema integrity — tables, columns, constraints match expected definitions after agent writes
- FR-09.2: Validate agent state persistence — agent memory written to PostgreSQL is correctly retrievable across sessions
- FR-09.3: Validate Redis cache correctness — cached values match source of truth, TTL behavior correct
- FR-09.4: Validate Redis cache invalidation — stale entries are evicted correctly when upstream data changes
- FR-09.5: Detect silent write failures — failed DB writes must surface, not silently drop data
- FR-09.6: Validate transaction integrity — multi-step agent operations that write to PostgreSQL must be atomic

**Acceptance criteria:**
- Schema validation passes on every deployment
- Zero silent write failures in test suite
- Redis TTL behavior confirmed on 10 defined expiry scenarios
- Transaction rollback confirmed on injected failure scenarios

---

### FR-10: Infrastructure and CI/CD Validation
**Priority:** High
**Covers:** Docker, GitHub Actions, GCP Cloud Run, GCE, Firebase

- FR-10.1: Validate Docker container health — containers start correctly, pass health checks, expose correct ports
- FR-10.2: Validate GCP Cloud Run service deployment — containerized agents respond correctly after deployment
- FR-10.3: Validate Firebase data layer — reads and writes from agent layer to Firebase behave correctly
- FR-10.4: Validate GitHub Actions CI/CD pipeline — all AgentProof test suites run correctly in CI environment
- FR-10.5: Validate cross-cloud behavior — data written on GCP is correctly readable by Azure services (Cosmos DB, AI Search) where applicable
- FR-10.6: Validate graceful degradation under infrastructure failure — Cloud Run scale-down, cold start latency within SLA
- FR-10.7: Prompt regression suite runs automatically on schedule (every 6 hours) via GitHub Actions

**Acceptance criteria:**
- Container health checks pass on every build
- Cloud Run cold start latency < 10 seconds
- CI/CD pipeline completes in < 10 minutes for unit + regression suite
- Prompt regression suite runs on schedule with zero manual intervention
- Cross-cloud read/write consistency confirmed on defined test scenarios

---

### FR-11: Nango Connector Validation
**Priority:** High
**Covers:** Nango 700+ connector integration layer (CRM, email, calendar, financial systems, documents)

- FR-11.1: Validate connector reliability — data flows correctly from each connected system through Nango
- FR-11.2: Detect silent connector failures — dropped or malformed records must surface, not disappear silently
- FR-11.3: Validate data completeness post-ingestion — all expected records present and correctly field-mapped
- FR-11.4: Validate schema drift handling — when a connected system updates its field structure, ingestion must handle gracefully or alert
- FR-11.5: Validate connector authentication — OAuth token refresh and credential rotation behave correctly
- FR-11.6: Validate rate limit handling — Nango connectors must back off correctly under source system rate limits
- FR-11.7: Validate audit trail continuity through connector layer — every ingested record traceable back to source

**Acceptance criteria:**
- Data completeness > 99.9% on test dataset across 5 connector types
- Silent failure detection rate 100% on injected connector failures
- Schema drift handling confirmed on 10 defined drift scenarios
- Auth token refresh confirmed without data loss
- Full audit trail confirmed from source system to knowledge graph

### NFR-01: Performance
- All evaluator methods must be async
- Parallel test execution supported via pytest-asyncio and pytest-xdist
- Single evaluation run completes in < 5 minutes for standard test suite
- Framework adds < 10% overhead to evaluated system latency during testing

### NFR-02: Reliability
- Evaluators must fail gracefully — one failing check must not abort the pipeline
- All external API calls must use exponential backoff retry logic
- Framework must run correctly when external services (Arthur, Qdrant, etc.) are unavailable (mock mode)

### NFR-03: Auditability
- Every evaluation run must produce an immutable JSONL audit log
- Logs must include: timestamp, model, input, output, metrics, scores, pass/fail, threshold used
- SHA-256 hash on every log entry for tamper evidence

### NFR-04: Accuracy
- Documentation must reflect actual implementation status at all times
- No component may be described as complete unless it has passing tests
- README and CLAUDE.md updated after every significant change

### NFR-05: Portability
- Python 3.11 only
- Runs locally, in Docker, and in GitHub Actions CI/CD
- No vendor lock-in — model providers, vector stores, and graph DBs are pluggable

### NFR-06: Security
- No API keys in code or logs — environment variables only
- PII detection on all inputs before logging
- Prompt injection detection on all external inputs

---

## Technical Constraints

| Constraint | Rule |
|---|---|
| LangChain | NEVER — zero imports, zero dependencies |
| LangGraph | NEVER — zero imports, zero dependencies |
| LangSmith | NEVER — use Langfuse or custom logging |
| Python version | 3.11 only |
| Async | All evaluator methods must be async |
| Return types | Typed dataclasses only — no raw dicts |
| Dependencies | Every dependency must exist on PyPI |
| Tests | Every component must have passing tests |

---

## Dependencies (v1.0)

```toml
[tool.poetry.dependencies]
python = "^3.11"
deepeval = "^1.0"
litellm = "^1.0"
anthropic = "^0.20"
openai = "^1.0"
qdrant-client = "^1.7"
asyncpg = "^0.29"              # PostgreSQL async driver
redis = "^5.0"                 # Redis async client
neo4j = "^5.0"                 # Neo4j driver (Memgraph-compatible)
fastapi = "^0.110"
pydantic = "^2.0"
pytest = "^8.0"
pytest-asyncio = "^0.23"
pytest-xdist = "^3.5"
httpx = "^0.27"
python-dotenv = "^1.0"
typer = "^0.12"
rich = "^13.0"
prometheus-client = "^0.20"
docker = "^7.0"                # Docker SDK for container validation
google-cloud-run = "^0.10"    # GCP Cloud Run client
firebase-admin = "^6.0"       # Firebase validation

[tool.poetry.group.dev.dependencies]
black = "^24.0"
ruff = "^0.4"
mypy = "^1.9"
hypothesis = "^6.100"
pytest-cov = "^5.0"
```
