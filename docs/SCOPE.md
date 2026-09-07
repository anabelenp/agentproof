# AgentProof — Scope Document
> Version 1.0 | May 2026 | ThinkAstra Consulting

---

## What AgentProof Is

AgentProof is an AI systems reliability and evaluation platform built for organizations deploying agentic AI systems in production. It is designed for problems assertion-based tooling cannot solve: non-deterministic LLM outputs, multi-model routing validation, knowledge graph integrity, retrieval quality, and governance audit trail verification.

AgentProof is built by Ana Bruno (ThinkAstra Consulting), with 20+ years of enterprise systems experience across financial services, cybersecurity, and enterprise software.

---

## The Problem AgentProof Solves

Enterprise organizations are deploying AI agents that make real decisions — underwriting claims, surfacing executive findings, routing customer inquiries, processing financial data. Conventional software checks assume deterministic outputs. AI agents don't give you that.

The result: companies ship AI systems they cannot test, cannot audit, and cannot trust.

**Specific problems AgentProof addresses:**

- Multi-model routing systems (LiteLLM, custom routers) produce different outputs depending on which model handled the request — all must meet the same quality bar
- RAG pipelines retrieve context from vector databases — wrong retrieval means wrong answers, and no one catches it before it reaches an executive
- 92-agent orchestration layers must coordinate correctly — a failed handoff or missed human-override checkpoint is invisible to traditional monitoring
- Knowledge graphs (Memgraph, Neo4j, Qdrant) must correctly represent organizational reality — entity resolution failures corrupt every downstream decision
- Non-technical end users (CEOs, CFOs, operations managers) receive AI-generated findings — confusion and trust failure are quality failures, not just UX issues
- Regulated industries (insurance, finance, healthcare) require complete audit trails — every agent action must be traceable and tamper-evident

---

## What AgentProof Is NOT

- Not a general-purpose test automation framework (use Playwright, Pytest for that)
- Not a production monitoring or APM product (use Langfuse for LLM tracing). AgentProof does export Prometheus metrics and in-memory eval traces for test runs via `prometheus-client`.
- Not an LLM provider or agent builder
- Not a replacement for unit and integration testing — it sits on top of those layers
- Not LangChain-dependent — zero LangChain or LangGraph imports, ever

---

## Target Users

**Primary:** Applied AI Engineers, ML platform engineers, and reliability engineers at companies deploying agentic AI systems in regulated or high-stakes environments

**Secondary:** Forward Deployed Engineers and AI Solutions Engineers validating client deployments

**Tertiary:** ThinkAstra Consulting clients needing AI systems reliability and evaluation as a service

---

## Target Environments

- Multi-agent orchestration platforms (custom-built or commercial)
- RAG pipelines backed by Qdrant, Pinecone, Weaviate, or similar vector databases
- Knowledge graph systems (Memgraph, Neo4j) and graph-adjacent vector stores (Qdrant)
- Relational and caching layers (PostgreSQL, Redis) used in agent memory and state management
- Multi-model routing layers (LiteLLM, custom routers)
- FastAPI-based agentic backends with async streaming responses
- Enterprise data ingestion pipelines via Nango (700+ connectors) and custom connectors
- Cloud infrastructure: Google Cloud Platform (Cloud Run, GCE, Firebase), Azure (Cosmos DB, AI Search)
- Container and CI/CD environments: Docker, GitHub Actions
- Regulated industries: insurance, financial services, healthcare, legal

---

## MVP Scope (v1.0 — 4 weeks)

### In scope:
- BaseEvaluator abstract class and unified test runner
- LiteLLM multi-model routing validator
- Output consistency tests across Claude + GPT
- DeepEval integration for LLM output scoring
- Qdrant retrieval quality evaluator
- PostgreSQL and Redis state/memory validation
- Entity resolution validator (Memgraph/Neo4j graph DB patterns)
- Governance audit trail integrity validator
- Streaming response and latency threshold tests
- Nango connector reliability and silent failure detection
- Docker-based local test environment
- GitHub Actions CI/CD pipeline with automated regression suite
- GCP Cloud Run and Firebase deployment validation patterns
- Immutable audit logging (JSONL + SHA-256)
- Clean README and demo-ready example suite

### Deferred to v1.1:
- Memgraph/Neo4j direct Cypher query validators (deep graph traversal)
- Azure Cosmos DB and AI Search validation
- Bias detection across agent outputs
- Dashboard and reporting UI
- Managed service / API layer
- Multi-tenant support

### Deferred to v2.0:
- AgentProof as a SaaS product (ThinkAstra offering)
- Browser-based test authoring
- Real-time monitoring integration
- Enterprise licensing model

---

## Success Criteria for MVP

1. Framework runs end-to-end with zero LangChain dependencies
2. LiteLLM routing validator catches model fallback failures reliably
3. DeepEval metrics produce scored output for relevance, faithfulness, hallucination
4. Qdrant retrieval evaluator validates ground truth recall above defined thresholds
5. Audit logger produces tamper-evident JSONL output for every evaluation run
6. All components have passing tests
7. README accurately describes current implementation status
8. Demo-ready: can be shown in a 30-minute technical interview without hitting stubs

---

## Constraints

- Python 3.11 only
- No LangChain, LangGraph, or LangSmith — hard constraint, never negotiable
- All evaluator methods must be async
- Typed return values throughout — no raw dicts
- Every component must have a corresponding test
- Documentation must reflect actual implementation status — no aspirational claims
