"""LLMEvaluator — DeepEval-backed scoring of LLM outputs.

Wraps AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, and
ToxicityMetric with
AgentProof's config, retry, audit, and ValidationResult contracts.

Usage:
    evaluator = LLMEvaluator(config, audit_logger)
    result = await evaluator.evaluate_relevance(
        input="What is our refund policy?",
        output="Refunds are available within 30 days of purchase.",
    )

    # Or dispatch via the BaseEvaluator interface:
    result = await evaluator.evaluate(
        input="What is our refund policy?",
        output="Refunds are available within 30 days of purchase.",
        metric="relevance",
    )
"""

from datetime import datetime, timezone
from typing import Any, Callable

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, LLMEvaluatorError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async
from agentproof.integrations.anthropic import AnthropicIntegration

SUPPORTED_METRICS = (
    "relevance",
    "answer_relevancy",
    "answer_relevance",
    "faithfulness",
    "hallucination",
    "toxicity",
)


class LLMEvaluator(BaseEvaluator):
    """Scores LLM outputs with DeepEval metrics and writes an audit entry.

    Four public metric methods plus `evaluate()` (the BaseEvaluator contract)
    and `evaluate_all()` (runs relevance, faithfulness, hallucination, toxicity).

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with thresholds, judge model, and retry settings.
        audit_logger: AuditLogger that receives every result.
        anthropic: Optional AnthropicIntegration used to generate `output`
            when the caller does not supply one.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        anthropic: AnthropicIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._anthropic = anthropic
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "LLMEvaluator"

    async def evaluate(
        self,
        *,
        input: str,
        output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        metric: str = "relevance",
        executive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a single DeepEval metric.

        Args:
            input: User query / prompt under evaluation.
            output: Model response. Generated via AnthropicIntegration if omitted.
            context: Ground-truth documents (hallucination) or retrieved docs
                (faithfulness fallback).
            retrieval_context: Retrieved documents for faithfulness. Preferred
                over `context` when both are supplied.
            metric: One of "relevance" (default), "faithfulness", "hallucination",
                "toxicity". "answer_relevancy" and "answer_relevance" alias relevance.
            executive: When True, faithfulness uses the executive threshold (0.95).

        Returns:
            ValidationResult for the requested metric. Never raises on
            evaluation failure; AuditError still propagates.
        """
        try:
            resolved_output = await self._resolve_output(input, output)
        except AuditError:
            raise
        except Exception as exc:
            audit_id = self._new_audit_id()
            start = self._start_timer()
            result = self._error_result(audit_id, start, metric, exc, threshold=0.0)
            await self._write_audit(result, model=self.config.judge_model)
            return result

        key = metric.lower().strip()
        if key in {"relevance", "answer_relevancy", "answer_relevance"}:
            return await self.evaluate_relevance(input=input, output=resolved_output)
        if key == "faithfulness":
            return await self.evaluate_faithfulness(
                input=input,
                output=resolved_output,
                context=context,
                retrieval_context=retrieval_context,
                executive=executive,
            )
        if key == "hallucination":
            return await self.evaluate_hallucination(
                input=input,
                output=resolved_output,
                context=context,
            )
        if key == "toxicity":
            return await self.evaluate_toxicity(input=input, output=resolved_output)

        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=metric,
            threshold=0.0,
            details={"supported_metrics": list(SUPPORTED_METRICS)},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
            error=(
                f"Unknown metric {metric!r}. "
                f"Expected one of: {', '.join(SUPPORTED_METRICS)}"
            ),
        )
        await self._write_audit(result, model=self.config.judge_model)
        return result

    async def evaluate_relevance(
        self,
        *,
        input: str,
        output: str,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether `output` answers `input` (AnswerRelevancyMetric).

        Args:
            input: User query.
            output: Model response.

        Returns:
            ValidationResult. `passed` is True when score >= relevance threshold
            (default 0.7). Higher is better.
        """
        def build_case() -> LLMTestCase:
            self._require_text(input, "input", "relevance")
            self._require_text(output, "output", "relevance")
            return LLMTestCase(input=input, actual_output=output)

        return await self._measure(
            metric_name="relevance",
            threshold=self.config.default_relevance_threshold,
            factory=AnswerRelevancyMetric,
            test_case_factory=build_case,
        )

    async def evaluate_faithfulness(
        self,
        *,
        input: str,
        output: str,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        executive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether `output` is grounded in provided documents.

        Args:
            input: User query.
            output: Model response.
            context: Fallback documents if `retrieval_context` is not supplied.
            retrieval_context: Documents the generator actually used (RAG).
            executive: If True, use `executive_faithfulness_threshold` (0.95).

        Returns:
            ValidationResult. `passed` is True when score >= faithfulness
            threshold (default 0.9). Higher is better.
        """
        threshold = (
            self.config.executive_faithfulness_threshold
            if executive
            else self.config.faithfulness_threshold
        )

        def build_case() -> LLMTestCase:
            self._require_text(input, "input", "faithfulness")
            self._require_text(output, "output", "faithfulness")
            docs = retrieval_context if retrieval_context is not None else context
            if not docs:
                raise LLMEvaluatorError(
                    "faithfulness requires non-empty context",
                    evaluator_name=self.name,
                    metric="faithfulness",
                )
            return LLMTestCase(
                input=input,
                actual_output=output,
                retrieval_context=list(docs),
            )

        return await self._measure(
            metric_name="faithfulness",
            threshold=threshold,
            factory=FaithfulnessMetric,
            test_case_factory=build_case,
        )

    async def evaluate_hallucination(
        self,
        *,
        input: str,
        output: str,
        context: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score unsupported claims against ground-truth `context`.

        AgentProof reports hallucination as a *rate* (lower is better) using
        `config.hallucination_threshold` (default 0.1). DeepEval 4.x's
        HallucinationMetric is higher-is-better (factual alignment), so this
        method converts: `rate = 1.0 - deepeval_score`.

        Args:
            input: User query.
            output: Model response.
            context: Ground-truth reference documents. Required.

        Returns:
            ValidationResult. `score` is the hallucination rate. `passed` is
            True when rate <= hallucination threshold.
        """
        def build_case() -> LLMTestCase:
            self._require_text(input, "input", "hallucination")
            self._require_text(output, "output", "hallucination")
            if not context:
                raise LLMEvaluatorError(
                    "hallucination requires non-empty context",
                    evaluator_name=self.name,
                    metric="hallucination",
                )
            return LLMTestCase(
                input=input,
                actual_output=output,
                context=list(context),
            )

        return await self._measure(
            metric_name="hallucination",
            threshold=self.config.hallucination_threshold,
            factory=HallucinationMetric,
            test_case_factory=build_case,
            invert_score=True,
        )

    async def evaluate_toxicity(
        self,
        *,
        input: str,
        output: str,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score toxic content in `output` (DeepEval ToxicityMetric).

        DeepEval's toxicity score is already a rate (lower is better).
        `passed` when score <= `config.toxicity_threshold` (default 0.1).

        Args:
            input: User query (required by DeepEval's test case).
            output: Model response under inspection.

        Returns:
            ValidationResult. `score` is the toxicity rate.
        """

        def build_case() -> LLMTestCase:
            self._require_text(input, "input", "toxicity")
            self._require_text(output, "output", "toxicity")
            return LLMTestCase(input=input, actual_output=output)

        return await self._measure(
            metric_name="toxicity",
            threshold=self.config.toxicity_threshold,
            factory=ToxicityMetric,
            test_case_factory=build_case,
            lower_is_better=True,
        )

    async def evaluate_all(
        self,
        *,
        input: str,
        output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        executive: bool = False,
    ) -> list[ValidationResult]:
        """Run relevance, faithfulness, hallucination, and toxicity in that order.

        Generates `output` once (if omitted) and reuses it for all three metrics.

        Args:
            input: User query.
            output: Model response. Generated via AnthropicIntegration if omitted.
            context: Documents for hallucination and as faithfulness fallback.
            retrieval_context: Retrieved documents for faithfulness.
            executive: Forwarded to evaluate_faithfulness.

        Returns:
            Four ValidationResult objects, one per metric. Individual metric
            failures are captured as error results; they do not abort the rest.
        """
        try:
            resolved_output = await self._resolve_output(input, output)
        except AuditError:
            raise
        except Exception as exc:
            audit_id = self._new_audit_id()
            start = self._start_timer()
            result = self._error_result(audit_id, start, "all", exc, threshold=0.0)
            await self._write_audit(result, model=self.config.judge_model)
            return [result]

        return [
            await self.evaluate_relevance(input=input, output=resolved_output),
            await self.evaluate_faithfulness(
                input=input,
                output=resolved_output,
                context=context,
                retrieval_context=retrieval_context,
                executive=executive,
            ),
            await self.evaluate_hallucination(
                input=input,
                output=resolved_output,
                context=context,
            ),
            await self.evaluate_toxicity(input=input, output=resolved_output),
        ]

    async def _resolve_output(self, prompt: str, output: str | None) -> str:
        """Return caller-supplied output or generate it via Anthropic.

        Args:
            prompt: User query used as the generation prompt when output is None.
            output: Caller-supplied model response, or None to generate.

        Returns:
            Non-empty output string.

        Raises:
            LLMEvaluatorError: Output is empty, or generation is requested
                without an AnthropicIntegration.
        """
        if output is not None:
            self._require_text(output, "output", "generation")
            return output
        if self._anthropic is None:
            raise LLMEvaluatorError(
                "output is required when AnthropicIntegration is not configured",
                evaluator_name=self.name,
                metric="generation",
            )
        return await self._anthropic.complete(prompt)

    async def _measure(
        self,
        *,
        metric_name: str,
        threshold: float,
        factory: Callable[..., Any],
        test_case_factory: Callable[[], LLMTestCase],
        invert_score: bool = False,
        lower_is_better: bool = False,
    ) -> ValidationResult:
        """Construct a DeepEval metric, measure asynchronously, and audit.

        Args:
            metric_name: Stored on ValidationResult.metric.
            threshold: AgentProof pass/fail threshold (pre-inversion).
            factory: DeepEval metric class.
            test_case_factory: Builds the LLMTestCase; may raise LLMEvaluatorError.
            invert_score: If True, treat DeepEval's score as higher-is-better
                alignment and convert to a lower-is-better rate.
            lower_is_better: If True, pass when score <= threshold (toxicity).

        Returns:
            ValidationResult. Evaluation exceptions become error results.
            AuditError propagates.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            test_case = test_case_factory()
            deepeval_threshold = (1.0 - threshold) if invert_score else threshold
            metric = factory(
                threshold=deepeval_threshold,
                model=self._judge_model(),
                include_reason=True,
            )

            async def _run() -> Any:
                return await metric.a_measure(test_case)

            await retry_async(self._retry)(_run)()

            raw = metric.score
            if raw is None:
                raise LLMEvaluatorError(
                    "DeepEval returned no score",
                    evaluator_name=self.name,
                    metric=metric_name,
                )
            raw_f = _clamp(float(raw))
            if invert_score:
                score = _clamp(1.0 - raw_f)
                passed = score <= threshold
                interpretation = "hallucination_rate (lower is better)"
            elif lower_is_better:
                score = raw_f
                passed = score <= threshold
                interpretation = "lower is better"
            else:
                score = raw_f
                passed = score >= threshold
                interpretation = "higher is better"

            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric=metric_name,
                threshold=threshold,
                details={
                    "reason": getattr(metric, "reason", None) or "",
                    "deepeval_score": raw_f,
                    "deepeval_success": bool(getattr(metric, "success", passed)),
                    "score_interpretation": interpretation,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except RetryExhaustedError as exc:
            cause = exc.last_exception or exc
            wrapped = LLMEvaluatorError(
                f"failed after {self._retry.max_attempts} attempts: "
                f"{type(cause).__name__}: {cause}",
                evaluator_name=self.name,
                metric=metric_name,
            )
            result = self._error_result(
                audit_id, start, metric_name, wrapped, threshold=threshold
            )
        except Exception as exc:
            result = self._error_result(
                audit_id, start, metric_name, exc, threshold=threshold
            )

        await self._write_audit(result, model=self.config.judge_model)
        return result

    def _judge_model(self) -> str | AnthropicModel:
        """Return the DeepEval judge: AnthropicModel for Claude, else a name string.

        Args:
            None — uses `config.judge_model` and `config.anthropic_api_key`.

        Returns:
            An AnthropicModel when the configured judge is a Claude model
            and `anthropic_api_key` is set; otherwise the model name string.
        """
        name = self.config.judge_model
        if name.startswith("claude") and self.config.anthropic_api_key:
            return AnthropicModel(
                model=name,
                api_key=self.config.anthropic_api_key,
                temperature=0,
            )
        return name

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise LLMEvaluatorError if `value` is empty or whitespace.

        Args:
            value: Field value to validate.
            field: Field name for the error message.
            metric: Metric being evaluated, stored on the error.

        Raises:
            LLMEvaluatorError: If value is empty or whitespace-only.
        """
        if not isinstance(value, str) or not value.strip():
            raise LLMEvaluatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0].

    Args:
        score: Raw numeric score.

    Returns:
        Score bounded to [0.0, 1.0].
    """
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
