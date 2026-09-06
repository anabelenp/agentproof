"""Unit tests for the AgentProof exception hierarchy."""

import pytest
from agentproof.core.errors import (
    AgentProofError,
    AuditError,
    ConfigurationError,
    DataLayerValidatorError,
    EvaluatorError,
    GovernanceValidatorError,
    IntegrationError,
    LLMEvaluatorError,
    RAGEvaluatorError,
    RetryExhaustedError,
    RoutingValidatorError,
    StreamingValidatorError,
)

# ── Hierarchy ─────────────────────────────────────────────────────────────────


def test_all_errors_inherit_from_base():
    for cls in [
        ConfigurationError,
        EvaluatorError,
        AuditError,
        RetryExhaustedError,
        IntegrationError,
    ]:
        assert issubclass(cls, AgentProofError)


def test_evaluator_subclasses_inherit_from_evaluator_error():
    for cls in [
        LLMEvaluatorError,
        RAGEvaluatorError,
        RoutingValidatorError,
        StreamingValidatorError,
        GovernanceValidatorError,
        DataLayerValidatorError,
    ]:
        assert issubclass(cls, EvaluatorError)


def test_all_errors_are_exceptions():
    for cls in [
        AgentProofError,
        ConfigurationError,
        EvaluatorError,
        AuditError,
        RetryExhaustedError,
        IntegrationError,
    ]:
        assert issubclass(cls, Exception)


def test_audit_error_is_not_evaluator_error():
    assert not issubclass(AuditError, EvaluatorError)


# ── EvaluatorError fields ─────────────────────────────────────────────────────


def test_evaluator_error_stores_fields():
    err = EvaluatorError("something broke", evaluator_name="LLMEvaluator", metric="faithfulness")
    assert err.evaluator_name == "LLMEvaluator"
    assert err.metric == "faithfulness"
    assert str(err) == "something broke"


def test_evaluator_error_default_fields():
    err = EvaluatorError("plain error")
    assert err.evaluator_name == ""
    assert err.metric == ""


def test_llm_evaluator_error_inherits_fields():
    err = LLMEvaluatorError("llm failed", evaluator_name="LLMEvaluator", metric="relevance")
    assert err.evaluator_name == "LLMEvaluator"
    assert err.metric == "relevance"


# ── RetryExhaustedError ───────────────────────────────────────────────────────


def test_retry_exhausted_captures_last_exception():
    cause = ValueError("root cause")
    err = RetryExhaustedError("gave up after 3 attempts", last_exception=cause)
    assert err.last_exception is cause
    assert "gave up" in str(err)


def test_retry_exhausted_last_exception_defaults_none():
    err = RetryExhaustedError("no attempts")
    assert err.last_exception is None


# ── Raise / catch ─────────────────────────────────────────────────────────────


def test_can_catch_subclass_as_base():
    with pytest.raises(AgentProofError):
        raise LLMEvaluatorError("test")


def test_can_catch_evaluator_error_as_evaluator():
    with pytest.raises(EvaluatorError):
        raise RAGEvaluatorError("rag failed")


def test_audit_error_not_caught_as_evaluator_error():
    with pytest.raises(AuditError):
        try:
            raise AuditError("audit failed")
        except EvaluatorError:
            pytest.fail("AuditError should not be caught as EvaluatorError")
