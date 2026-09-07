"""AgentProof exception hierarchy.

All exceptions inherit from AgentProofError.
The only exception that always propagates (never swallowed) is AuditError —
a failed audit write is a compliance violation, not a recoverable condition.

Usage:
    raise ConfigurationError("ANTHROPIC_API_KEY is required")
    raise LLMEvaluatorError("DeepEval call failed", evaluator_name="LLMEvaluator", metric="faithfulness")
"""


class AgentProofError(Exception):
    """Base exception for all AgentProof errors."""


class ConfigurationError(AgentProofError):
    """Invalid or missing configuration value."""


class EvaluatorError(AgentProofError):
    """An evaluator failed to execute.

    Does not abort the pipeline — captured in ValidationResult.error instead.

    Args:
        message: Human-readable description of the failure.
        evaluator_name: Name of the evaluator that raised this error.
        metric: The metric being evaluated when the error occurred.
    """

    def __init__(
        self,
        message: str,
        evaluator_name: str = "",
        metric: str = "",
    ) -> None:
        super().__init__(message)
        self.evaluator_name = evaluator_name
        self.metric = metric


class LLMEvaluatorError(EvaluatorError):
    """LLM output evaluation failure (DeepEval, judge model, etc.)."""


class RAGEvaluatorError(EvaluatorError):
    """RAG pipeline evaluation failure (retrieval or generation stage)."""


class RoutingValidatorError(EvaluatorError):
    """Multi-model routing validation failure."""


class StreamingValidatorError(EvaluatorError):
    """Streaming response validation failure (TTFT, throughput, completeness)."""


class GovernanceValidatorError(EvaluatorError):
    """Governance audit trail validation failure."""


class DataLayerValidatorError(EvaluatorError):
    """PostgreSQL / Redis data-layer validation failure."""


class GuardrailValidatorError(EvaluatorError):
    """PII, injection, policy, or tool-allowlist guardrail failure."""


class WorkflowEvaluatorError(EvaluatorError):
    """Agentic workflow trace evaluation failure."""


class GraphValidatorError(EvaluatorError):
    """Knowledge-graph integrity validation failure (Neo4j / Memgraph)."""


class AuditError(AgentProofError):
    """Audit logging failure.

    ALWAYS propagates — never caught or swallowed.
    A failed audit write is a compliance violation, not a recoverable condition.
    """


class RetryExhaustedError(AgentProofError):
    """All retry attempts exhausted without success.

    Args:
        message: Description including function name and attempt count.
        last_exception: The exception raised on the final attempt.
    """

    def __init__(
        self,
        message: str,
        last_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.last_exception = last_exception


class IntegrationError(AgentProofError):
    """External service integration failure (network, auth, schema mismatch)."""
