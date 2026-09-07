"""GuardrailValidator — PII, prompt injection, policy, and tool allowlists.

Deterministic checks that fail closed. These do not call an LLM so they can
run on every input/output before audit logging or tool dispatch.

Usage:
    guard = GuardrailValidator(config, audit_logger)
    result = await guard.validate_pii(text="Contact jane@acme.com")
    result = await guard.validate_injection(text=user_prompt)
    result = await guard.validate_tool_allowlist(
        requested=["qdrant.search"],
        allowed=["qdrant.search", "postgres.fetch"],
    )
"""

from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, GuardrailValidatorError
from agentproof.core.safety import detect_injection, detect_pii, redact_pii

SUPPORTED_METRICS = (
    "pii",
    "prompt_injection",
    "policy",
    "tool_allowlist",
)


class GuardrailValidator(BaseEvaluator):
    """Blocks unsafe inputs, outputs, and tool calls.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig (pii_redaction controls audit redaction).
        audit_logger: AuditLogger that receives every result.
    """

    def __init__(self, config: AgentProofConfig, audit_logger: AuditLogger) -> None:
        super().__init__(config, audit_logger)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "GuardrailValidator"

    async def evaluate(
        self,
        *,
        metric: str = "pii",
        text: str = "",
        input: str = "",
        output: str = "",
        forbidden_patterns: list[str] | None = None,
        requested: list[str] | None = None,
        allowed: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a guardrail check.

        Args:
            metric: pii (default), prompt_injection, policy, tool_allowlist.
            text: Payload for pii / injection / policy. Falls back to
                `input` then `output`.
            input / output: Alternate payload fields.
            forbidden_patterns: Substrings or regex-like phrases for policy.
            requested: Tool / MCP names the agent attempted.
            allowed: Allowlist of tool / MCP names.

        Returns:
            ValidationResult for the requested metric.
        """
        payload = text or input or output
        key = metric.lower().strip()
        if key == "pii":
            return await self.validate_pii(text=payload)
        if key in {"prompt_injection", "injection"}:
            return await self.validate_injection(text=payload)
        if key == "policy":
            return await self.validate_policy(
                text=payload, forbidden_patterns=forbidden_patterns or []
            )
        if key in {"tool_allowlist", "mcp_allowlist"}:
            return await self.validate_tool_allowlist(
                requested=requested or [],
                allowed=allowed or [],
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

    async def validate_pii(self, *, text: str, **kwargs: Any) -> ValidationResult:
        """Fail when emails, SSNs, phones, cards, or API keys are present.

        Args:
            text: Input or output to scan.

        Returns:
            ValidationResult. score is 1.0 when clean, else 0.0. Findings
            report kind only — matched spans are redacted.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(text, "text", "pii")
            findings = detect_pii(text)
            kinds = sorted({item.kind for item in findings})
            passed = not findings
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="pii",
                threshold=1.0,
                details={
                    "finding_count": len(findings),
                    "kinds": kinds,
                    "redacted": redact_pii(text),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(audit_id, start, "pii", exc, threshold=1.0)

        await self._write_audit(result)
        return result

    async def validate_injection(self, *, text: str, **kwargs: Any) -> ValidationResult:
        """Fail when classic jailbreak / prompt-injection phrases appear.

        Args:
            text: User prompt or tool argument.

        Returns:
            ValidationResult. score is 1.0 when clean.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(text, "text", "prompt_injection")
            findings = detect_injection(text)
            kinds = sorted({item.kind for item in findings})
            passed = not findings
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="prompt_injection",
                threshold=1.0,
                details={
                    "finding_count": len(findings),
                    "kinds": kinds,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "prompt_injection", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_policy(
        self,
        *,
        text: str,
        forbidden_patterns: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Fail when `text` contains any forbidden phrase (case-insensitive).

        Args:
            text: Output or tool argument.
            forbidden_patterns: Phrases that must not appear.

        Returns:
            ValidationResult. score is clean-phrases / total-phrases.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(text, "text", "policy")
            if not forbidden_patterns:
                raise GuardrailValidatorError(
                    "policy requires non-empty forbidden_patterns",
                    evaluator_name=self.name,
                    metric="policy",
                )
            haystack = text.lower()
            hits = [
                pattern
                for pattern in forbidden_patterns
                if pattern and pattern.lower() in haystack
            ]
            total = len(forbidden_patterns)
            score = (total - len(hits)) / total if total else 1.0
            result = ValidationResult(
                passed=not hits,
                score=score if 0.0 <= score <= 1.0 else 0.0,
                evaluator_name=self.name,
                metric="policy",
                threshold=1.0,
                details={
                    "forbidden_patterns": list(forbidden_patterns),
                    "hits": hits,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(audit_id, start, "policy", exc, threshold=1.0)

        await self._write_audit(result)
        return result

    async def validate_tool_allowlist(
        self,
        *,
        requested: list[str],
        allowed: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Fail when any requested MCP / tool name is outside `allowed`.

        Args:
            requested: Tool names the agent attempted (e.g. `github.create_pr`).
            allowed: Explicit allowlist. Empty allowlist fails closed.

        Returns:
            ValidationResult. score is allowed-requests / total-requests.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not allowed:
                raise GuardrailValidatorError(
                    "tool_allowlist requires a non-empty allowed list",
                    evaluator_name=self.name,
                    metric="tool_allowlist",
                )
            permitted = {name.strip() for name in allowed if name.strip()}
            names = [name.strip() for name in requested if name.strip()]
            denied = [name for name in names if name not in permitted]
            total = len(names)
            allowed_count = total - len(denied)
            score = (allowed_count / total) if total else 1.0
            result = ValidationResult(
                passed=not denied,
                score=score if 0.0 <= score <= 1.0 else 0.0,
                evaluator_name=self.name,
                metric="tool_allowlist",
                threshold=1.0,
                details={
                    "requested": names,
                    "allowed": sorted(permitted),
                    "denied": denied,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "tool_allowlist", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        text: str = "",
        forbidden_patterns: list[str] | None = None,
        requested: list[str] | None = None,
        allowed: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Run PII and injection, plus policy/allowlist when arguments exist.

        Args:
            text: Payload for pii / injection / policy.
            forbidden_patterns: If set, also run policy.
            requested / allowed: If allowed is set, also run tool_allowlist.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results = [
            await self.validate_pii(text=text),
            await self.validate_injection(text=text),
        ]
        if forbidden_patterns:
            results.append(
                await self.validate_policy(
                    text=text, forbidden_patterns=forbidden_patterns
                )
            )
        if allowed is not None:
            results.append(
                await self.validate_tool_allowlist(
                    requested=requested or [],
                    allowed=allowed,
                )
            )
        return results

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise GuardrailValidatorError if `value` is empty or whitespace."""
        if not isinstance(value, str) or not value.strip():
            raise GuardrailValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )
