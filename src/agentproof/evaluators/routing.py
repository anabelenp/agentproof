"""RoutingValidator — LiteLLM multi-model routing, fallback, and consistency.

Three DESIGN test types plus cost-tier routing:

1. Routing correctness — did the serving model match `expected_model`?
2. Fallback — when primary fails, does fallback serve within SLA?
3. Output consistency — same prompt across models meets the quality bar
   with score delta below `routing_consistency_delta`.
4. Cost routing — low-complexity tasks must not land on high-cost models.

Usage:
    validator = RoutingValidator(config, audit_logger)
    result = await validator.validate_routing_correctness(
        prompt="Summarize this claim.",
        expected_model="claude-sonnet-4-6",
    )
"""

from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, RoutingValidatorError
from agentproof.evaluators.llm import LLMEvaluator
from agentproof.integrations.litellm import (
    LiteLLMIntegration,
    is_high_cost_model,
    models_match,
)

SUPPORTED_METRICS = (
    "routing_correctness",
    "fallback",
    "output_consistency",
    "cost_routing",
)


class RoutingValidator(BaseEvaluator):
    """Validates multi-model routing behavior through LiteLLM.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with fallback SLA and consistency delta.
        audit_logger: AuditLogger that receives every result.
        litellm: Optional LiteLLMIntegration. Created from config if omitted.
        llm: Optional LLMEvaluator used for output-consistency scoring.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        litellm: LiteLLMIntegration | None = None,
        llm: LLMEvaluator | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._litellm = litellm or LiteLLMIntegration(config)
        self._llm = llm

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "RoutingValidator"

    @property
    def llm(self) -> LLMEvaluator:
        """LLMEvaluator used to score outputs for consistency checks."""
        if self._llm is None:
            self._llm = LLMEvaluator(self.config, self._audit)
        return self._llm

    @property
    def integration(self) -> LiteLLMIntegration:
        """The LiteLLMIntegration used for completions."""
        return self._litellm

    async def evaluate(
        self,
        *,
        prompt: str | None = None,
        input: str | None = None,
        expected_model: str | None = None,
        model: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        models: list[str] | None = None,
        metric: str = "routing_correctness",
        consistency_metric: str = "relevance",
        expected_tier: str = "low",
        forbidden_models: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a routing validation method.

        Args:
            prompt: User prompt. Alias of `input`.
            input: User prompt (TestRunner-friendly name).
            expected_model: Model that should serve (routing_correctness).
            model: Router alias / model to call. Defaults to expected_model.
            primary_model: Primary model for fallback tests.
            fallback_model: Secondary model for fallback tests.
            models: Models to compare for output_consistency.
            metric: One of routing_correctness (default), fallback,
                output_consistency, cost_routing.
            consistency_metric: LLMEvaluator metric for consistency scoring.
            expected_tier: "low" or "high" for cost_routing.
            forbidden_models: Extra denylist for cost_routing.

        Returns:
            ValidationResult for the requested metric.
        """
        resolved_prompt = prompt if prompt is not None else (input or "")
        key = metric.lower().strip()

        if key == "routing_correctness":
            return await self.validate_routing_correctness(
                prompt=resolved_prompt,
                expected_model=expected_model or model or "",
                model=model,
            )
        if key == "fallback":
            return await self.validate_fallback(
                prompt=resolved_prompt,
                primary_model=primary_model or model or "",
                fallback_model=fallback_model or "",
            )
        if key == "output_consistency":
            return await self.validate_output_consistency(
                prompt=resolved_prompt,
                models=models or [],
                metric=consistency_metric,
            )
        if key == "cost_routing":
            return await self.validate_cost_routing(
                prompt=resolved_prompt,
                model=model or expected_model or "",
                expected_tier=expected_tier,
                forbidden_models=forbidden_models,
            )

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
        await self._write_audit(result)
        return result

    async def validate_routing_correctness(
        self,
        *,
        prompt: str,
        expected_model: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm the serving model matches `expected_model`.

        Catches silent routing failures: HTTP 200 with the wrong model.

        Args:
            prompt: User prompt.
            expected_model: Model that should have handled the request.
            model: Router alias to call. Defaults to expected_model.

        Returns:
            ValidationResult. score is 1.0 on match, 0.0 otherwise.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        requested = model or expected_model
        try:
            self._require_text(prompt, "prompt", "routing_correctness")
            self._require_text(expected_model, "expected_model", "routing_correctness")
            completion = await self._litellm.complete(prompt, model=requested)
            matched = models_match(completion.model, expected_model)
            result = ValidationResult(
                passed=matched,
                score=1.0 if matched else 0.0,
                evaluator_name=self.name,
                metric="routing_correctness",
                threshold=1.0,
                details={
                    "requested_model": requested,
                    "expected_model": expected_model,
                    "actual_model": completion.model,
                    "matched": matched,
                    "completion_latency_ms": completion.latency_ms,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "routing_correctness", exc, threshold=1.0
            )

        await self._write_audit(result, model=requested)
        return result

    async def validate_fallback(
        self,
        *,
        prompt: str,
        primary_model: str,
        fallback_model: str,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm fallback activates when the primary model fails.

        Passes when fallback served, the serving model matches
        `fallback_model`, and total failover time is within
        `config.max_fallback_seconds` (default 5s).

        If the primary succeeds, the result fails — the fallback path was
        not exercised.

        Args:
            prompt: User prompt.
            primary_model: Model expected to fail.
            fallback_model: Model that must serve after primary failure.

        Returns:
            ValidationResult. score is 1.0 on a successful failover inside SLA.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        sla_s = self.config.max_fallback_seconds
        try:
            self._require_text(prompt, "prompt", "fallback")
            self._require_text(primary_model, "primary_model", "fallback")
            self._require_text(fallback_model, "fallback_model", "fallback")
            completion = await self._litellm.complete(
                prompt,
                model=primary_model,
                fallback_model=fallback_model,
            )
            elapsed_s = completion.latency_ms / 1000.0
            within_sla = elapsed_s <= sla_s
            served_fallback = models_match(completion.model, fallback_model)
            activated = completion.used_fallback
            passed = activated and served_fallback and within_sla
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="fallback",
                threshold=0.0,
                details={
                    "primary_model": primary_model,
                    "fallback_model": fallback_model,
                    "actual_model": completion.model,
                    "fallback_activated": activated,
                    "served_fallback": served_fallback,
                    "failover_seconds": elapsed_s,
                    "sla_seconds": sla_s,
                    "within_sla": within_sla,
                    "primary_error": completion.primary_error,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "fallback", exc, threshold=0.0
            )

        await self._write_audit(result, model=fallback_model)
        return result

    async def validate_output_consistency(
        self,
        *,
        prompt: str,
        models: list[str],
        metric: str = "relevance",
        **kwargs: Any,
    ) -> ValidationResult:
        """Score the same prompt across models and bound the quality delta.

        Each output is scored with LLMEvaluator (default: relevance). Passes
        when every score meets the metric threshold and
        max(score) - min(score) <= `config.routing_consistency_delta`.

        Args:
            prompt: User prompt.
            models: Two or more model names to compare.
            metric: LLMEvaluator metric name (relevance, faithfulness, ...).

        Returns:
            ValidationResult. score is 1.0 - delta (higher = more consistent).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        delta_threshold = self.config.routing_consistency_delta
        quality_threshold = self.config.default_relevance_threshold
        try:
            self._require_text(prompt, "prompt", "output_consistency")
            if len(models) < 2:
                raise RoutingValidatorError(
                    "output_consistency requires at least two models",
                    evaluator_name=self.name,
                    metric="output_consistency",
                )

            per_model: list[dict[str, Any]] = []
            scores: list[float] = []
            quality_failures: list[str] = []
            scoring_errors: list[str] = []

            for model_name in models:
                completion = await self._litellm.complete(prompt, model=model_name)
                scored = await self.llm.evaluate(
                    input=prompt, output=completion.text, metric=metric
                )
                entry: dict[str, Any] = {
                    "requested_model": model_name,
                    "actual_model": completion.model,
                    "score": scored.score,
                    "passed": scored.passed,
                    "error": scored.error,
                }
                per_model.append(entry)
                if scored.error:
                    scoring_errors.append(f"{model_name}: {scored.error}")
                    continue
                scores.append(scored.score)
                if not scored.passed:
                    quality_failures.append(model_name)

            if scoring_errors:
                raise RoutingValidatorError(
                    "LLMEvaluator failed for one or more models: "
                    + "; ".join(scoring_errors),
                    evaluator_name=self.name,
                    metric="output_consistency",
                )
            if not scores:
                raise RoutingValidatorError(
                    "output_consistency produced no scores",
                    evaluator_name=self.name,
                    metric="output_consistency",
                )

            delta = max(scores) - min(scores)
            delta_ok = delta <= delta_threshold
            quality_ok = not quality_failures
            passed = quality_ok and delta_ok
            result = ValidationResult(
                passed=passed,
                score=_clamp(1.0 - delta),
                evaluator_name=self.name,
                metric="output_consistency",
                threshold=delta_threshold,
                details={
                    "models": list(models),
                    "per_model": per_model,
                    "scores": scores,
                    "delta": delta,
                    "delta_threshold": delta_threshold,
                    "quality_threshold": quality_threshold,
                    "quality_failures": quality_failures,
                    "consistency_metric": metric,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "output_consistency", exc, threshold=delta_threshold
            )

        await self._write_audit(result)
        return result

    async def validate_cost_routing(
        self,
        *,
        prompt: str,
        model: str,
        expected_tier: str = "low",
        forbidden_models: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a prompt was not routed to a disallowed high-cost model.

        When `expected_tier` is "low", the serving model must not be a
        high-cost frontier model (opus / gpt-4 / o1 / o3, excluding mini
        variants) and must not appear in `forbidden_models`.

        Args:
            prompt: User prompt (typically a low-complexity task).
            model: Router alias to call.
            expected_tier: "low" (default) or "high".
            forbidden_models: Extra denylist of model names.

        Returns:
            ValidationResult. score is 1.0 when the cost tier is respected.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        denied = [m for m in (forbidden_models or [])]
        try:
            self._require_text(prompt, "prompt", "cost_routing")
            self._require_text(model, "model", "cost_routing")
            completion = await self._litellm.complete(prompt, model=model)
            actual = completion.model
            high_cost = is_high_cost_model(actual)
            forbidden_hit = any(models_match(actual, item) for item in denied)
            tier = expected_tier.lower().strip() or "low"
            if tier == "low":
                passed = (not high_cost) and (not forbidden_hit)
            else:
                passed = high_cost and (not forbidden_hit)
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="cost_routing",
                threshold=1.0,
                details={
                    "requested_model": model,
                    "actual_model": actual,
                    "expected_tier": tier,
                    "is_high_cost": high_cost,
                    "forbidden_models": denied,
                    "forbidden_hit": forbidden_hit,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "cost_routing", exc, threshold=1.0
            )

        await self._write_audit(result, model=model)
        return result

    async def evaluate_all(
        self,
        *,
        prompt: str,
        expected_model: str,
        primary_model: str,
        fallback_model: str,
        models: list[str],
    ) -> list[ValidationResult]:
        """Run correctness, fallback, and consistency in that order.

        Args:
            prompt: User prompt shared by all three checks.
            expected_model: Expected serving model for correctness.
            primary_model: Primary model for the fallback check.
            fallback_model: Fallback model for the fallback check.
            models: Models compared for output consistency.

        Returns:
            Three ValidationResult objects. Individual failures do not abort
            the rest.
        """
        return [
            await self.validate_routing_correctness(
                prompt=prompt, expected_model=expected_model
            ),
            await self.validate_fallback(
                prompt=prompt,
                primary_model=primary_model,
                fallback_model=fallback_model,
            ),
            await self.validate_output_consistency(prompt=prompt, models=models),
        ]

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise RoutingValidatorError if `value` is empty or whitespace.

        Args:
            value: Field value to validate.
            field: Field name for the error message.
            metric: Metric being evaluated, stored on the error.

        Raises:
            RoutingValidatorError: If value is empty or whitespace-only.
        """
        if not isinstance(value, str) or not value.strip():
            raise RoutingValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
