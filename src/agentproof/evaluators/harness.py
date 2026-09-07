"""EvalHarness — suite runner with guardrails, traces, and Prometheus metrics.

Wraps TestRunner so a list of EvalCase objects can be scored by any
BaseEvaluator, optionally gated by GuardrailValidator, with an EvalTrace
and MetricsRegistry recorded for every run.

Usage:
    harness = EvalHarness(config)
    harness.add_evaluator(LLMEvaluator(config, harness.audit_logger), metric="relevance")
    harness.add_case(EvalCase(case_id="refund", input="...", output="..."))
    summary, trace = await harness.run()
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator
from agentproof.core.config import AgentProofConfig
from agentproof.core.observability import (
    EvalTrace,
    MetricsRegistry,
    TraceStore,
    span_from_result,
)
from agentproof.core.runner import TestRunner, TestRunSummary
from agentproof.validators.guardrails import GuardrailValidator


@dataclass
class EvalCase:
    """One row in an evaluation suite.

    Args:
        case_id: Stable id used in traces and details.
        input: User prompt / query.
        output: Model (or agent) response. Optional if the evaluator generates it.
        context: Ground-truth documents.
        retrieval_context: Retrieved documents for RAG metrics.
        metric: Default metric forwarded to the evaluator.
        tags: Free-form labels for filtering reports.
        extra: Extra kwargs merged into evaluate() (trace, expected_agents, …).
    """

    case_id: str
    input: str = ""
    output: str = ""
    context: list[str] = field(default_factory=list)
    retrieval_context: list[str] = field(default_factory=list)
    metric: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        """Build evaluate() kwargs, omitting empty optional fields.

        Returns:
            Kwargs suitable for BaseEvaluator.evaluate.
        """
        kwargs: dict[str, Any] = {"input": self.input}
        if self.output:
            kwargs["output"] = self.output
        if self.context:
            kwargs["context"] = list(self.context)
        if self.retrieval_context:
            kwargs["retrieval_context"] = list(self.retrieval_context)
        if self.metric:
            kwargs["metric"] = self.metric
        kwargs.update(self.extra)
        return kwargs


class EvalHarness:
    """Runs eval cases through registered evaluators plus optional guardrails.

    Args:
        config: AgentProofConfig.
        audit_logger: Optional logger. Created from config when omitted.
        metrics: Prometheus registry. Created when omitted.
        traces: In-memory TraceStore. Created when omitted.
        guardrails: Optional GuardrailValidator run on each case text first.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger | None = None,
        metrics: MetricsRegistry | None = None,
        traces: TraceStore | None = None,
        guardrails: GuardrailValidator | None = None,
    ) -> None:
        self.config = config
        self.audit_logger = audit_logger or AuditLogger(
            config.audit_log_dir, redact_pii=config.pii_redaction
        )
        self.metrics = metrics or MetricsRegistry()
        self.traces = traces or TraceStore()
        self.guardrails = guardrails
        self._evaluators: list[tuple[BaseEvaluator, dict[str, Any]]] = []
        self._cases: list[EvalCase] = []

    def add_evaluator(self, evaluator: BaseEvaluator, **kwargs: Any) -> None:
        """Register an evaluator applied to every case.

        Args:
            evaluator: Any BaseEvaluator.
            **kwargs: Defaults merged under each case's kwargs (case wins).
        """
        self._evaluators.append((evaluator, kwargs))

    def add_case(self, case: EvalCase) -> None:
        """Append a suite row.

        Args:
            case: EvalCase to run.
        """
        self._cases.append(case)

    async def run(self, *, parallel: bool = False) -> tuple[TestRunSummary, EvalTrace]:
        """Execute guardrails (if set) then every evaluator on every case.

        Sequential by default so case order is preserved in the trace.

        Args:
            parallel: Forwarded to TestRunner.run.

        Returns:
            (TestRunSummary, EvalTrace) for this harness run.
        """
        runner = TestRunner(
            self.config,
            audit_logger=self.audit_logger,
            metrics=self.metrics,
            traces=self.traces,
        )
        for case in self._cases:
            text = case.input or case.output
            if self.guardrails is not None and text:
                runner.register(
                    self.guardrails,
                    metric="pii",
                    text=text,
                    case_id=case.case_id,
                )
                runner.register(
                    self.guardrails,
                    metric="prompt_injection",
                    text=text,
                    case_id=case.case_id,
                )
            for evaluator, defaults in self._evaluators:
                kwargs = dict(defaults)
                kwargs.update(case.as_kwargs())
                kwargs.setdefault("metric", defaults.get("metric", case.metric))
                runner.register(evaluator, **kwargs)
        summary = await runner.run(parallel=parallel)
        stored = self.traces.list()
        trace = stored[-1] if stored else EvalTrace(
            trace_id=str(uuid4()),
            run_id=summary.run_id,
            spans=[span_from_result(result) for result in summary.results],
        )
        return summary, trace
