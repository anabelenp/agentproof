"""TestRunner — orchestrates multiple evaluators into a single test run.

Evaluators are registered with their kwargs, then run in parallel (default)
or sequentially. One failing or erroring evaluator never aborts the others.
Results are aggregated into a TestRunSummary with pass/fail counts and timing.

Usage:
    runner = TestRunner(config)
    runner.register(LLMEvaluator(config, audit_logger), input="...", output="...")
    runner.register(RAGEvaluator(config, audit_logger), query="...", context=[...])
    summary = await runner.run()
    sys.exit(0 if summary.all_passed else 1)
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .audit import AuditLogger
from .base import BaseEvaluator, ValidationResult
from .config import AgentProofConfig
from .observability import EvalTrace, MetricsRegistry, TraceStore, span_from_result


@dataclass
class TestRunSummary:
    """Aggregated results from a single TestRunner.run() call.

    Attributes:
        run_id: UUID for this specific run (useful for correlating audit entries).
        timestamp: UTC datetime when the run started.
        total: Total number of evaluators executed.
        passed: Evaluators that returned passed=True with no error.
        failed: Evaluators that returned passed=False with no error.
        error_count: Evaluators that raised or set result.error.
        duration_ms: Total wall-clock time for the run in milliseconds.
        results: Ordered list of ValidationResult objects (parallel order not guaranteed).
    """

    run_id: str
    timestamp: datetime
    total: int
    passed: int
    failed: int
    error_count: int
    duration_ms: float
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of evaluations that passed, ignoring errors. Returns 0.0 if total==0."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        """True only when every evaluator passed and no errors occurred."""
        return self.failed == 0 and self.error_count == 0


class TestRunner:
    """Orchestrates multiple BaseEvaluator instances into a test run.

    Creates a shared AuditLogger from config.audit_log_dir so all evaluators
    registered here write to the same daily JSONL file.

    Usage:
        runner = TestRunner(config)
        runner.register(my_evaluator, input="query", output="response")
        summary = await runner.run(parallel=True)
    """

    def __init__(
        self,
        config: AgentProofConfig,
        *,
        audit_logger: AuditLogger | None = None,
        metrics: MetricsRegistry | None = None,
        traces: TraceStore | None = None,
    ) -> None:
        """
        Args:
            config: AgentProofConfig. audit_log_dir is used for the shared AuditLogger.
            audit_logger: Optional logger. Created from config when omitted.
            metrics: Optional Prometheus registry. Created when omitted.
            traces: Optional in-memory TraceStore. Created when omitted.
        """
        self.config = config
        self.audit_logger = audit_logger or AuditLogger(
            config.audit_log_dir, redact_pii=config.pii_redaction
        )
        self.metrics = metrics or MetricsRegistry()
        self.traces = traces or TraceStore()
        self._evaluators: list[tuple[BaseEvaluator, dict[str, Any]]] = []

    def register(self, evaluator: BaseEvaluator, **kwargs: Any) -> None:
        """Register an evaluator with the kwargs to pass to evaluate().

        Args:
            evaluator: A BaseEvaluator subclass instance.
            **kwargs: Forwarded verbatim to evaluator.evaluate(**kwargs).
        """
        self._evaluators.append((evaluator, kwargs))

    async def _run_one(
        self,
        evaluator: BaseEvaluator,
        kwargs: dict[str, Any],
    ) -> ValidationResult:
        """Execute one evaluator, converting unexpected exceptions to error results.

        Well-behaved evaluators catch their own exceptions and return ValidationResult
        with error set. This is the safety net for unexpected failures.

        Args:
            evaluator: The evaluator to run.
            kwargs: Kwargs passed to evaluator.evaluate().

        Returns:
            ValidationResult — never raises (except AuditError, which propagates).
        """
        try:
            return await evaluator.evaluate(**kwargs)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                score=0.0,
                evaluator_name=evaluator.name,
                metric="unknown",
                threshold=0.0,
                details={},
                latency_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                audit_id=str(uuid.uuid4()),
                error=f"Uncaught exception in {evaluator.name!r}: {type(exc).__name__}: {exc}",
            )

    async def run(self, *, parallel: bool = True) -> TestRunSummary:
        """Execute all registered evaluators and return an aggregated summary.

        Args:
            parallel: If True (default), run all evaluators concurrently via
                      asyncio.gather. If False, run sequentially in registration order.

        Returns:
            TestRunSummary with pass/fail/error counts and all ValidationResults.
        """
        import time

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        start = time.perf_counter()

        if parallel:
            results: list[ValidationResult] = list(
                await asyncio.gather(
                    *[self._run_one(ev, kw) for ev, kw in self._evaluators]
                )
            )
        else:
            results = []
            for ev, kw in self._evaluators:
                results.append(await self._run_one(ev, kw))

        duration_ms = (time.perf_counter() - start) * 1000.0

        passed = sum(1 for r in results if r.passed and r.error is None)
        failed = sum(1 for r in results if not r.passed and r.error is None)
        error_count = sum(1 for r in results if r.error is not None)

        for result in results:
            self.metrics.record_result(result)
        self.traces.record(
            EvalTrace(
                trace_id=str(uuid.uuid4()),
                run_id=run_id,
                timestamp=timestamp,
                spans=[span_from_result(result) for result in results],
            )
        )

        return TestRunSummary(
            run_id=run_id,
            timestamp=timestamp,
            total=len(results),
            passed=passed,
            failed=failed,
            error_count=error_count,
            duration_ms=duration_ms,
            results=results,
        )
