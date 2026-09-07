"""Unit tests for Prometheus metrics, traces, EvalHarness, and audit redaction."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agentproof.core.audit import AuditEntry, AuditLogger
from agentproof.core.base import ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.observability import MetricsRegistry, TraceStore, span_from_result
from agentproof.core.runner import TestRunner
from agentproof.evaluators.harness import EvalCase, EvalHarness
from agentproof.evaluators.llm import LLMEvaluator


def _result(*, passed: bool = True, error: str | None = None) -> ValidationResult:
    return ValidationResult(
        passed=passed,
        score=0.9 if passed else 0.2,
        evaluator_name="LLMEvaluator",
        metric="relevance",
        threshold=0.7,
        details={"note": "ok"},
        latency_ms=12.0,
        timestamp=datetime.now(timezone.utc),
        audit_id=str(uuid4()),
        error=error,
    )


def test_metrics_record_and_export():
    metrics = MetricsRegistry()
    metrics.record_result(_result())
    text = metrics.export()
    assert "agentproof_evaluations_total" in text
    assert 'outcome="passed"' in text
    assert "agentproof_evaluation_latency_ms" in text


def test_guardrail_block_counter():
    metrics = MetricsRegistry()
    blocked = _result(passed=False)
    blocked.evaluator_name = "GuardrailValidator"
    blocked.metric = "pii"
    metrics.record_result(blocked)
    assert "agentproof_guardrail_blocks_total" in metrics.export()


def test_trace_store_get_and_overflow():
    store = TraceStore(max_traces=2)
    from agentproof.core.observability import EvalTrace, Span

    store.record(EvalTrace(trace_id="a", run_id="r", spans=[]))
    store.record(EvalTrace(trace_id="b", run_id="r", spans=[]))
    store.record(EvalTrace(trace_id="c", run_id="r", spans=[Span("x", "evaluation", 1.0, "ok")]))
    assert store.get("a") is None
    assert store.get("c") is not None
    assert store.get("c").status == "ok"


def test_span_from_result_error_status():
    span = span_from_result(_result(error="boom"))
    assert span.status == "error"
    assert span.kind == "evaluation"


async def test_runner_records_metrics_and_trace(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    metrics = MetricsRegistry()
    traces = TraceStore()
    runner = TestRunner(config, metrics=metrics, traces=traces)
    ev = MagicMock()
    ev.name = "LLMEvaluator"
    ev.evaluate = AsyncMock(return_value=_result())
    runner.register(ev, input="q", output="a")
    summary = await runner.run()
    assert summary.passed == 1
    assert "agentproof_evaluations_total" in metrics.export()
    assert len(traces.list()) == 1
    assert traces.list()[0].spans[0].status == "ok"


async def test_harness_runs_case_and_guardrails(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    harness = EvalHarness(config)
    evaluator = LLMEvaluator(config, harness.audit_logger)
    harness.add_evaluator(evaluator, metric="relevance")
    harness.add_case(
        EvalCase(case_id="refund", input="What is the refund policy?", output="30 days.")
    )
    cls = MagicMock()
    inst = MagicMock()
    inst.score = 0.88
    inst.reason = "ok"
    inst.success = True
    inst.a_measure = AsyncMock(return_value=0.88)
    cls.return_value = inst
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        summary, trace = await harness.run()
    assert summary.total == 1
    assert summary.passed == 1
    assert trace.run_id == summary.run_id


async def test_audit_redacts_pii_in_details(tmp_path):
    logger = AuditLogger(tmp_path / "audit_logs", redact_pii=True)
    entry = AuditEntry(
        audit_id=str(uuid4()),
        evaluator="LLMEvaluator",
        metric="relevance",
        score=0.9,
        threshold=0.7,
        passed=True,
        model="",
        latency_ms=1.0,
        details={"note": "email jane@acme.com"},
    )
    await logger.log(entry)
    assert "jane@acme.com" not in entry.details["note"]
    rows = await logger.read_entries()
    assert "jane@acme.com" not in rows[0]["details"]["note"]
    assert logger.verify_entry_data(rows[0]) is True
