"""BaseEvaluator and ValidationResult — the core contracts for all evaluators.

Every evaluator in AgentProof:
  1. Inherits from BaseEvaluator.
  2. Implements name (str property) and evaluate(**kwargs) -> ValidationResult.
  3. Calls self._write_audit(result) before returning.
  4. Never raises on evaluation failure — sets result.error instead.

Every evaluation produces exactly one ValidationResult and one AuditEntry.

Usage:
    class MyEvaluator(BaseEvaluator):
        @property
        def name(self) -> str:
            return "my_evaluator"

        async def evaluate(self, *, input: str, output: str, **kwargs) -> ValidationResult:
            audit_id = self._new_audit_id()
            start = self._start_timer()
            try:
                score = await self._score_somehow(input, output)
                result = ValidationResult(
                    passed=score >= self.config.default_relevance_threshold,
                    score=score,
                    evaluator_name=self.name,
                    metric="my_metric",
                    threshold=self.config.default_relevance_threshold,
                    details={"reason": "..."},
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
            except Exception as exc:
                result = self._error_result(audit_id, start, "my_metric", exc)
            await self._write_audit(result)
            return result
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .audit import AuditEntry, AuditLogger, new_audit_id
from .config import AgentProofConfig
from .errors import AuditError


@dataclass
class ValidationResult:
    """The canonical return type for every AgentProof evaluation.

    Never return raw dicts from evaluators. Always return this.

    Attributes:
        passed: True if score >= threshold.
        score: Normalized evaluation score in [0.0, 1.0].
        evaluator_name: Unique name of the evaluator (matches BaseEvaluator.name).
        metric: The specific metric evaluated (e.g. "faithfulness", "relevance").
        threshold: The pass/fail threshold applied to score.
        details: Metric-specific context — reasoning text, sub-scores, etc.
        latency_ms: Wall-clock evaluation time in milliseconds.
        timestamp: UTC datetime when the evaluation completed.
        audit_id: UUID linking this result to its audit log entry.
        error: Set on evaluation failure; None on success. Never raise — set this.
    """

    passed: bool
    score: float
    evaluator_name: str
    metric: str
    threshold: float
    details: dict[str, Any]
    latency_ms: float
    timestamp: datetime
    audit_id: str
    error: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"ValidationResult.score must be in [0.0, 1.0], got {self.score!r}"
            )


class BaseEvaluator(ABC):
    """Abstract base for all AgentProof evaluators.

    Subclasses must implement:
      - name (property) — unique identifier used in audit logs and reports.
      - evaluate(**kwargs) — async evaluation returning a ValidationResult.

    evaluate() MUST call self._write_audit(result) before returning.
    evaluate() MUST NOT raise on evaluation failure — set result.error instead.
    AuditError is the one exception that is allowed to propagate.

    Usage:
        evaluator = MyEvaluator(config, audit_logger)
        result = await evaluator.evaluate(input="...", output="...")
    """

    def __init__(self, config: AgentProofConfig, audit_logger: AuditLogger) -> None:
        """
        Args:
            config: AgentProofConfig with thresholds, model settings, and API keys.
            audit_logger: AuditLogger instance that writes to the JSONL audit trail.
        """
        self.config = config
        self._audit = audit_logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique evaluator identifier. Used in audit logs and reports."""

    @abstractmethod
    async def evaluate(self, **kwargs: Any) -> ValidationResult:
        """Run the evaluation and return a ValidationResult.

        Contract:
          - Always call self._write_audit(result) before returning.
          - On failure, set result.error — do not raise.
          - Always be async (never blocking).

        Returns:
            ValidationResult with audit_id linked to the audit log entry.

        Raises:
            AuditError: If the audit write fails. This always propagates.
        """

    async def _write_audit(
        self,
        result: ValidationResult,
        *,
        model: str = "",
    ) -> None:
        """Write a ValidationResult to the audit log.

        Args:
            result: The ValidationResult to log. audit_id must be pre-set.
            model: The LLM model name used in this evaluation (optional).

        Raises:
            AuditError: On audit write failure. Always propagates.
        """
        entry = AuditEntry(
            audit_id=result.audit_id,
            evaluator=result.evaluator_name,
            metric=result.metric,
            score=result.score,
            threshold=result.threshold,
            passed=result.passed,
            model=model,
            latency_ms=result.latency_ms,
            details=result.details,
            error=result.error,
        )
        await self._audit.log(entry)

    def _new_audit_id(self) -> str:
        """Generate a UUID v4 audit ID to share between ValidationResult and AuditEntry."""
        return new_audit_id()

    @staticmethod
    def _start_timer() -> float:
        """Return a perf_counter start time for latency measurement."""
        return time.perf_counter()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Compute milliseconds elapsed since a _start_timer() call.

        Args:
            start: Return value of _start_timer().

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.perf_counter() - start) * 1000.0

    def _error_result(
        self,
        audit_id: str,
        start: float,
        metric: str,
        exc: Exception,
        threshold: float = 0.0,
    ) -> ValidationResult:
        """Build a failed ValidationResult from an exception.

        Used inside evaluate() catch blocks to avoid duplication.

        Args:
            audit_id: Pre-generated audit ID for this evaluation attempt.
            start: Timer start from _start_timer().
            metric: The metric that was being evaluated.
            exc: The exception that caused the failure.
            threshold: The threshold that was in use (default 0.0 if unknown).

        Returns:
            ValidationResult with passed=False, score=0.0, and error set.
        """
        return ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=metric,
            threshold=threshold,
            details={},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
            error=f"{type(exc).__name__}: {exc}",
        )
