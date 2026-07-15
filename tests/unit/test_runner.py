"""Unit tests for TestRunner and TestRunSummary."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.runner import TestRunSummary, TestRunner


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_result(
    name: str,
    *,
    passed: bool = True,
    score: float = 0.9,
    error: str | None = None,
) -> ValidationResult:
    return ValidationResult(
        passed=passed,
        score=score,
        evaluator_name=name,
        metric="test_metric",
        threshold=0.7,
        details={},
        latency_ms=10.0,
        timestamp=datetime.now(timezone.utc),
        audit_id=str(uuid.uuid4()),
        error=error,
    )


def mock_evaluator(
    name: str,
    *,
    passed: bool = True,
    score: float = 0.9,
    error: str | None = None,
    raises: Exception | None = None,
) -> BaseEvaluator:
    ev = MagicMock(spec=BaseEvaluator)
    ev.name = name
    if raises is not None:
        ev.evaluate = AsyncMock(side_effect=raises)
    else:
        ev.evaluate = AsyncMock(return_value=make_result(name, passed=passed, score=score, error=error))
    return ev


@pytest.fixture
def config() -> AgentProofConfig:
    return AgentProofConfig()


@pytest.fixture
def runner(config) -> TestRunner:
    return TestRunner(config)


# ── Empty run ─────────────────────────────────────────────────────────────────


async def test_empty_run_returns_zero_counts(runner):
    summary = await runner.run()
    assert summary.total == 0
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.error_count == 0
    assert summary.all_passed is True


async def test_empty_run_pass_rate_is_zero(runner):
    summary = await runner.run()
    assert summary.pass_rate == 0.0


# ── Single evaluator ──────────────────────────────────────────────────────────


async def test_single_passing_evaluator(runner):
    runner.register(mock_evaluator("a", passed=True))
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.all_passed is True


async def test_single_failing_evaluator(runner):
    runner.register(mock_evaluator("a", passed=False, score=0.4))
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 0
    assert summary.failed == 1
    assert summary.all_passed is False


# ── Multiple evaluators ───────────────────────────────────────────────────────


async def test_mixed_results(runner):
    runner.register(mock_evaluator("a", passed=True))
    runner.register(mock_evaluator("b", passed=True))
    runner.register(mock_evaluator("c", passed=False))
    summary = await runner.run()
    assert summary.total == 3
    assert summary.passed == 2
    assert summary.failed == 1
    assert summary.all_passed is False


async def test_pass_rate_calculation(runner):
    runner.register(mock_evaluator("a", passed=True))
    runner.register(mock_evaluator("b", passed=True))
    runner.register(mock_evaluator("c", passed=False))
    summary = await runner.run()
    assert abs(summary.pass_rate - 2 / 3) < 1e-6


async def test_results_list_length(runner):
    for i in range(5):
        runner.register(mock_evaluator(f"eval_{i}"))
    summary = await runner.run()
    assert len(summary.results) == 5


# ── Error handling ────────────────────────────────────────────────────────────


async def test_uncaught_exception_becomes_error_result(runner):
    runner.register(mock_evaluator("bad", raises=RuntimeError("exploded")))
    summary = await runner.run()
    assert summary.error_count == 1
    assert summary.passed == 0
    assert summary.all_passed is False
    assert summary.results[0].error is not None
    assert "RuntimeError" in summary.results[0].error


async def test_one_error_does_not_abort_others(runner):
    runner.register(mock_evaluator("bad", raises=RuntimeError("boom")))
    runner.register(mock_evaluator("good", passed=True))
    summary = await runner.run()
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.error_count == 1


async def test_evaluator_error_result_has_evaluator_name(runner):
    runner.register(mock_evaluator("named_eval", raises=RuntimeError("fail")))
    summary = await runner.run()
    assert summary.results[0].evaluator_name == "named_eval"


# ── Error vs failed distinction ───────────────────────────────────────────────


async def test_error_result_not_counted_as_failed(runner):
    runner.register(mock_evaluator("err", raises=RuntimeError("oops")))
    summary = await runner.run()
    assert summary.failed == 0
    assert summary.error_count == 1


async def test_error_field_set_counts_as_error_not_failed(runner):
    runner.register(mock_evaluator("e", passed=False, error="api timeout"))
    summary = await runner.run()
    assert summary.error_count == 1
    assert summary.failed == 0


# ── Sequential mode ───────────────────────────────────────────────────────────


async def test_sequential_mode_all_evaluators_run(runner):
    for i in range(3):
        runner.register(mock_evaluator(f"seq_{i}"))
    summary = await runner.run(parallel=False)
    assert summary.total == 3


async def test_sequential_mode_order_preserved(runner):
    call_order: list[str] = []

    async def record_a(**kwargs):
        call_order.append("a")
        return make_result("a")

    async def record_b(**kwargs):
        call_order.append("b")
        return make_result("b")

    ev_a = MagicMock(spec=BaseEvaluator)
    ev_a.name = "a"
    ev_a.evaluate = record_a

    ev_b = MagicMock(spec=BaseEvaluator)
    ev_b.name = "b"
    ev_b.evaluate = record_b

    runner.register(ev_a)
    runner.register(ev_b)
    await runner.run(parallel=False)

    assert call_order == ["a", "b"]


# ── TestRunSummary ────────────────────────────────────────────────────────────


async def test_summary_has_valid_run_id(runner):
    summary = await runner.run()
    parsed = uuid.UUID(summary.run_id)
    assert str(parsed) == summary.run_id


async def test_summary_has_timestamp(runner):
    summary = await runner.run()
    assert isinstance(summary.timestamp, datetime)


async def test_summary_duration_non_negative(runner):
    runner.register(mock_evaluator("a"))
    summary = await runner.run()
    assert summary.duration_ms >= 0.0


def test_all_passed_false_when_failed():
    summary = TestRunSummary(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        total=2,
        passed=1,
        failed=1,
        error_count=0,
        duration_ms=100.0,
    )
    assert summary.all_passed is False


def test_all_passed_false_when_error():
    summary = TestRunSummary(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        total=2,
        passed=1,
        failed=0,
        error_count=1,
        duration_ms=100.0,
    )
    assert summary.all_passed is False


# ── kwargs forwarding ─────────────────────────────────────────────────────────


async def test_kwargs_forwarded_to_evaluate(runner):
    ev = MagicMock(spec=BaseEvaluator)
    ev.name = "kwarg_test"
    ev.evaluate = AsyncMock(return_value=make_result("kwarg_test"))

    runner.register(ev, input="hello", output="world", extra="extra_val")
    await runner.run()

    ev.evaluate.assert_called_once_with(input="hello", output="world", extra="extra_val")
