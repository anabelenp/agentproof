# AgentProof Tutorial

## Engineering Reliable AI Systems: A Practical Guide to LLM, RAG, and Agent Evaluation

A practical guide to evaluating AI applications across retrieval, model routing, tool use, generation, data ingestion, governance, observability, and end-to-end agent behavior.

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

Traditional software systems generally have deterministic interfaces: given an input and a known program state, the expected behavior can often be expressed as an exact value or a well-defined set of conditions.

AI systems are fundamentally different.

A single user request can produce multiple responses that are semantically equivalent, while two responses that look similar can differ substantially in factual accuracy, grounding, safety, or usefulness.

Consider a customer service agent responding to:

> "Can I return this after 30 days?"

The agent might respond:

> "Our standard return window is 30 days, so you're right at the boundary — please contact support to check eligibility."

Or:

> "Returns are accepted within 30 days from purchase."

Or:

> "I'd recommend reaching out to our team as your situation is at the limit of our policy."

All three responses could be valid depending on the organization's policy and the surrounding context. They may share no common substring.

Evaluating them therefore requires understanding **meaning, evidence, intent, and behavior**, rather than comparing the response against a fixed string.

This is the problem AgentProof addresses.

AgentProof is an AI systems reliability and evaluation platform. It evaluates whether an AI system satisfies defined **behavioral contracts**.

Instead of asking only:

> "Does the output match the expected string?"

AgentProof asks questions such as:

- Is the response faithful to the source material?
- Did the system retrieve the right evidence?
- Does the response answer the user's actual question?
- Did the model introduce unsupported claims?
- Did the routing layer select the appropriate model?
- Did a fallback occur correctly when a provider failed?
- Did an ingestion pipeline preserve all source records?
- Did an agent use only authorized tools?
- Is the evaluation result auditable?
- Did the complete workflow satisfy its operational and governance requirements?

This makes AgentProof useful not only for evaluating model output, but for validating **AI systems as complete production systems**.

The framework provides:

- Typed evaluation result contracts
- Configurable behavioral thresholds
- LLM-as-a-Judge evaluation
- RAG retrieval evaluation
- Multi-model routing validation
- Streaming performance validation
- Data-layer validation
- Graph validation
- External-system ingestion validation
- Guardrail validation
- Workflow and agent-trace evaluation
- Prometheus-compatible observability
- Tamper-evident audit logging
- Retry handling for transient infrastructure failures

The architectural principle is simple:

> **AI reliability requires evaluating the entire system, not just the final response.**

---

## 2. The Tech Stack, Explained

### DeepEval

**What it is:** An open-source evaluation framework that uses language models as judges to evaluate language-model outputs. It provides metrics including answer relevancy, faithfulness, hallucination, contextual recall, contextual precision, and toxicity.

**Why AgentProof uses it:** AgentProof delegates semantic evaluation to DeepEval rather than implementing every LLM-as-a-Judge metric from scratch. This allows AgentProof to concentrate on orchestration, result contracts, auditability, thresholds, error handling, and system-level evaluation.

DeepEval provides the semantic scoring capability; AgentProof provides the surrounding evaluation infrastructure.

> ⚠️ **IMPORTANT:** LLM-as-a-Judge scores are probabilistic rather than deterministic. The judge model can produce slightly different scores across runs. AgentProof therefore uses configurable thresholds rather than exact score assertions and preserves evaluation history so quality can be analyzed over time.

---

### LiteLLM

**What it is:** A unified interface for interacting with multiple LLM providers through a common API.

**Why AgentProof uses it:** Enterprise AI applications frequently use multiple models for reasons including capability, cost, latency, availability, data residency, and compliance.

AgentProof uses LiteLLM to validate multi-model routing behavior without requiring separate application-level integrations for every provider.

For example:

```text
User request
     │
     ▼
Routing layer
     │
     ├──→ Model A — general requests
     ├──→ Model B — complex reasoning
     └──→ Model C — restricted-data workloads