"""Unit tests for ValidationResult and BaseEvaluator."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_result(**overrides) -> ValidationResult:
    defaults = dict(
        passed=True,
        score=0.85,
        evaluator_name="test_evaluator",
        metric="relevance",
        threshold=0.7,
        details={"reason": "test"},
        latency_ms=42.0,
        timestamp=datetime.now(timezone.utc),
        audit_id=str(uuid.uuid4()),
    )
    defaults.update(overrides)
    return ValidationResult(**defaults)


class ConcreteEvaluator(BaseEvaluator):
    """Minimal concrete implementation for testing BaseEvaluator contracts."""

    @property
    def name(self) -> str:
        return "concrete_evaluator"

    async def evaluate(self, *, input: str = "", output: str = "", **kwargs) -> ValidationResult:
        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=True,
            score=0.9,
            evaluator_name=self.name,
            metric="test_metric",
            threshold=self.config.default_relevance_threshold,
            details={"input_len": len(input), "output_len": len(output)},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
        )
        await self._write_audit(result, model="test-model")
        return result


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(mock_audit) -> ConcreteEvaluator:
    return ConcreteEvaluator(AgentProofConfig(), mock_audit)


# ── ValidationResult construction ─────────────────────────────────────────────


def test_validation_result_defaults():
    r = make_result()
    assert r.error is None
    assert r.passed is True
    assert r.score == 0.85


def test_validation_result_with_error():
    r = make_result(passed=False, score=0.0, error="API timeout")
    assert r.error == "API timeout"
    assert r.passed is False


def test_score_exactly_zero_is_valid():
    r = make_result(score=0.0, passed=False)
    assert r.score == 0.0


def test_score_exactly_one_is_valid():
    r = make_result(score=1.0)
    assert r.score == 1.0


def test_score_above_one_raises():
    with pytest.raises(ValueError, match="score must be in"):
        make_result(score=1.001)


def test_score_below_zero_raises():
    with pytest.raises(ValueError):
        make_result(score=-0.001)


# ── BaseEvaluator abstract interface ──────────────────────────────────────────


def test_cannot_instantiate_base_evaluator_directly():
    with pytest.raises(TypeError):
        BaseEvaluator(AgentProofConfig(), MagicMock())  # type: ignore[abstract]


def test_concrete_evaluator_has_name(evaluator):
    assert evaluator.name == "concrete_evaluator"


# ── evaluate() contract ───────────────────────────────────────────────────────


async def test_evaluate_returns_validation_result(evaluator):
    result = await evaluator.evaluate(input="hello", output="world")
    assert isinstance(result, ValidationResult)


async def test_evaluate_result_fields(evaluator):
    result = await evaluator.evaluate(input="hi", output="there")
    assert result.evaluator_name == "concrete_evaluator"
    assert result.metric == "test_metric"
    assert 0.0 <= result.score <= 1.0
    assert result.audit_id != ""
    assert result.latency_ms >= 0.0
    assert result.error is None


async def test_evaluate_calls_audit_log(evaluator, mock_audit):
    await evaluator.evaluate(input="x", output="y")
    mock_audit.log.assert_called_once()


async def test_audit_id_in_result_matches_audit_entry(evaluator, mock_audit):
    result = await evaluator.evaluate(input="x", output="y")
    logged_entry = mock_audit.log.call_args[0][0]
    assert logged_entry.audit_id == result.audit_id


# ── Helper methods ────────────────────────────────────────────────────────────


def test_new_audit_id_is_valid_uuid(evaluator):
    aid = evaluator._new_audit_id()
    parsed = uuid.UUID(aid)
    assert str(parsed) == aid


def test_elapsed_ms_is_non_negative(evaluator):
    start = evaluator._start_timer()
    elapsed = evaluator._elapsed_ms(start)
    assert elapsed >= 0.0


def test_error_result_structure(evaluator):
    start = evaluator._start_timer()
    audit_id = evaluator._new_audit_id()
    exc = RuntimeError("api call failed")
    result = evaluator._error_result(audit_id, start, "faithfulness", exc)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metric == "faithfulness"
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert "api call failed" in result.error
    assert result.audit_id == audit_id
