"""Prometheus metrics and in-memory eval traces.

AgentProof is a test harness, not a production APM. This module records
evaluation outcomes so CI, a local `/metrics` scrape, or a later reporter
can observe pass rate, latency, and guardrail blocks without LangSmith.

Usage:
    metrics = MetricsRegistry()
    metrics.record_result(result)
    payload = metrics.export()

    traces = TraceStore()
    traces.record(EvalTrace(trace_id="...", run_id="...", spans=[...]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from agentproof.core.base import ValidationResult

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

_LATENCY_BUCKETS = (10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0)
_SCORE_BUCKETS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def _outcome(result: ValidationResult) -> str:
    """Map a ValidationResult to a metrics outcome label."""
    if result.error:
        return "error"
    return "passed" if result.passed else "failed"


@dataclass
class Span:
    """One timed step inside an evaluation run.

    Args:
        name: Span name (evaluator or guardrail metric).
        kind: evaluation, guardrail, or workflow.
        duration_ms: Wall-clock duration.
        status: ok, fail, or error.
        attributes: Low-cardinality context (metric, agent, tool).
    """

    name: str
    kind: str
    duration_ms: float
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTrace:
    """Structured trace for one harness or TestRunner run.

    Args:
        trace_id: Unique id for this trace.
        run_id: TestRunner / harness run id.
        timestamp: UTC time the trace closed.
        spans: Ordered spans.
    """

    trace_id: str
    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    spans: list[Span] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Worst span status: error > fail > ok."""
        statuses = {span.status for span in self.spans}
        if "error" in statuses:
            return "error"
        if "fail" in statuses:
            return "fail"
        return "ok"


class TraceStore:
    """In-memory ring of EvalTrace objects for the current process.

    Args:
        max_traces: Oldest traces are dropped after this many entries.
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self._traces: list[EvalTrace] = []

    def record(self, trace: EvalTrace) -> None:
        """Append a trace, dropping the oldest when over capacity.

        Args:
            trace: Completed EvalTrace.
        """
        self._traces.append(trace)
        overflow = len(self._traces) - self.max_traces
        if overflow > 0:
            self._traces = self._traces[overflow:]

    def list(self) -> list[EvalTrace]:
        """Return traces oldest-first."""
        return list(self._traces)

    def get(self, trace_id: str) -> EvalTrace | None:
        """Look up a trace by id.

        Args:
            trace_id: Trace identifier.

        Returns:
            Matching EvalTrace, or None.
        """
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def clear(self) -> None:
        """Drop all stored traces."""
        self._traces.clear()


class MetricsRegistry:
    """Process-local Prometheus registry for evaluation telemetry.

    A dedicated CollectorRegistry keeps unit tests from mutating the global
    default registry.

    Args:
        registry: Optional existing CollectorRegistry.
        namespace: Prometheus metric namespace (default `agentproof`).
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        namespace: str = "agentproof",
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self.evaluations_total = Counter(
            f"{namespace}_evaluations_total",
            "Evaluation results by evaluator, metric, and outcome",
            labelnames=("evaluator", "metric", "outcome"),
            registry=self.registry,
        )
        self.evaluation_latency_ms = Histogram(
            f"{namespace}_evaluation_latency_ms",
            "Evaluation wall-clock latency in milliseconds",
            labelnames=("evaluator", "metric"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.evaluation_score = Histogram(
            f"{namespace}_evaluation_score",
            "Evaluation score distribution in [0, 1]",
            labelnames=("evaluator", "metric"),
            buckets=_SCORE_BUCKETS,
            registry=self.registry,
        )
        self.guardrail_blocks_total = Counter(
            f"{namespace}_guardrail_blocks_total",
            "Guardrail checks that blocked (passed=false, no error)",
            labelnames=("metric",),
            registry=self.registry,
        )

    def record_result(self, result: ValidationResult) -> None:
        """Increment counters and observe latency/score for one result.

        Args:
            result: Completed ValidationResult.
        """
        outcome = _outcome(result)
        labels = {
            "evaluator": result.evaluator_name or "unknown",
            "metric": result.metric or "unknown",
        }
        self.evaluations_total.labels(**labels, outcome=outcome).inc()
        self.evaluation_latency_ms.labels(**labels).observe(max(result.latency_ms, 0.0))
        self.evaluation_score.labels(**labels).observe(result.score)
        if (
            result.evaluator_name == "GuardrailValidator"
            and not result.passed
            and result.error is None
        ):
            self.guardrail_blocks_total.labels(metric=result.metric or "unknown").inc()

    def export(self) -> str:
        """Return Prometheus text exposition for this registry.

        Returns:
            Decoded `generate_latest` payload.
        """
        return generate_latest(self.registry).decode("utf-8")


def span_from_result(result: ValidationResult, *, kind: str = "evaluation") -> Span:
    """Build a Span from a ValidationResult.

    Args:
        result: Completed evaluation.
        kind: Span kind override.

    Returns:
        Span with status derived from passed/error.
    """
    if result.error:
        status = "error"
    elif result.passed:
        status = "ok"
    else:
        status = "fail"
    resolved_kind = kind
    if result.evaluator_name == "GuardrailValidator":
        resolved_kind = "guardrail"
    elif result.evaluator_name == "WorkflowEvaluator":
        resolved_kind = "workflow"
    return Span(
        name=f"{result.evaluator_name}.{result.metric}",
        kind=resolved_kind,
        duration_ms=result.latency_ms,
        status=status,
        attributes={
            "evaluator": result.evaluator_name,
            "metric": result.metric,
            "score": result.score,
            "audit_id": result.audit_id,
        },
    )
