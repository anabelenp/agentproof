"""GovernanceValidator — audit trail completeness, override, and separation.

Validates human-in-the-loop governance over agent runs:

1. Audit trail completeness — expected checkpoints are present, hashes verify.
2. Human override — an override is logged and downstream actions apply it.
3. Structural separation — agent A cannot perform actions reserved for agent B.
4. Tamper evidence — SHA-256 hashes on logger-sourced entries are intact.

Events may be passed in-memory (unit tests, live traces) or loaded from
`AuditLogger` JSONL files. Logger entries store run metadata in `details`.

Usage:
    validator = GovernanceValidator(config, audit_logger)
    result = await validator.validate_audit_trail_completeness(
        agent_run_id="run-1",
        expected_checkpoints=["retrieve", "draft", "human_review", "issue"],
        events=trace,
    )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, GovernanceValidatorError

SUPPORTED_METRICS = (
    "audit_trail_completeness",
    "human_override",
    "structural_separation",
    "tamper_evidence",
)


@dataclass
class OverrideEvent:
    """A human-in-the-loop override of an agent decision.

    Args:
        run_id: Agent run this override belongs to.
        checkpoint: Decision point that was overridden.
        agent: Agent whose action was overridden.
        original_action: Action the agent proposed.
        override_action: Action the human substituted.
        override_by: Identity of the human (or role).
        timestamp: Optional ISO-8601 timestamp.
    """

    run_id: str
    checkpoint: str
    agent: str
    original_action: str
    override_action: str
    override_by: str
    timestamp: str = ""


class GovernanceValidator(BaseEvaluator):
    """Validates governance audit trails for multi-agent runs.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig (audit_log_dir used when events are omitted).
        audit_logger: AuditLogger that receives every result and can supply
            historical JSONL entries.
    """

    def __init__(self, config: AgentProofConfig, audit_logger: AuditLogger) -> None:
        super().__init__(config, audit_logger)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "GovernanceValidator"

    async def evaluate(
        self,
        *,
        metric: str = "audit_trail_completeness",
        agent_run_id: str = "",
        expected_checkpoints: list[str] | None = None,
        events: list[dict[str, Any]] | None = None,
        override_event: OverrideEvent | dict[str, Any] | None = None,
        agent_a: str = "",
        agent_b: str = "",
        forbidden_actions: list[str] | None = None,
        require_hashes: bool | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a governance check.

        Args:
            metric: One of audit_trail_completeness (default), human_override,
                structural_separation, tamper_evidence.
            agent_run_id: Run id to filter events.
            expected_checkpoints: Required checkpoints for completeness.
            events: In-memory trail. Loaded from AuditLogger when omitted.
            override_event: OverrideEvent (or dict) for human_override.
            agent_a: Agent that must not perform forbidden_actions.
            agent_b: Agent that owns those actions (recorded in details).
            forbidden_actions: Actions reserved for agent_b.
            require_hashes: If True, every event must carry a valid SHA-256.
                Defaults to True when events are loaded from the logger.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        if key == "audit_trail_completeness":
            return await self.validate_audit_trail_completeness(
                agent_run_id=agent_run_id,
                expected_checkpoints=expected_checkpoints or [],
                events=events,
                require_hashes=require_hashes,
            )
        if key == "human_override":
            if override_event is None:
                audit_id = self._new_audit_id()
                start = self._start_timer()
                exc = GovernanceValidatorError(
                    "human_override requires override_event",
                    evaluator_name=self.name,
                    metric="human_override",
                )
                result = self._error_result(
                    audit_id, start, "human_override", exc, threshold=1.0
                )
                await self._write_audit(result)
                return result
            parsed = _coerce_override(override_event)
            return await self.validate_human_override(
                agent_run_id=agent_run_id or parsed.run_id,
                override_event=parsed,
                events=events,
            )
        if key == "structural_separation":
            return await self.validate_structural_separation(
                agent_a=agent_a,
                agent_b=agent_b,
                forbidden_actions=forbidden_actions or [],
                events=events,
                agent_run_id=agent_run_id,
            )
        if key == "tamper_evidence":
            return await self.validate_tamper_evidence(
                events=events,
                agent_run_id=agent_run_id,
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

    async def validate_audit_trail_completeness(
        self,
        *,
        agent_run_id: str,
        expected_checkpoints: list[str],
        events: list[dict[str, Any]] | None = None,
        require_hashes: bool | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm every expected checkpoint is present and hashes verify.

        Args:
            agent_run_id: Run id used to filter events.
            expected_checkpoints: Ordered checkpoint names that must appear.
            events: In-memory trail. Loaded from AuditLogger when omitted.
            require_hashes: Override hash requirement. Defaults to True when
                events are loaded from the logger.

        Returns:
            ValidationResult. score is found/expected. passed only when nothing
            is missing and no hashed entry fails integrity.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(agent_run_id, "agent_run_id", "audit_trail_completeness")
            if not expected_checkpoints:
                raise GovernanceValidatorError(
                    "expected_checkpoints must be a non-empty list",
                    evaluator_name=self.name,
                    metric="audit_trail_completeness",
                )
            loaded_from_logger = events is None
            normalized = await self._load_events(events, agent_run_id)
            must_hash = require_hashes if require_hashes is not None else loaded_from_logger

            present = [event["checkpoint"] for event in normalized if event["checkpoint"]]
            missing = [cp for cp in expected_checkpoints if cp not in present]
            order_ok = _checkpoints_in_order(present, expected_checkpoints)

            tampered: list[str] = []
            unhashed: list[str] = []
            for event in normalized:
                raw = event.get("raw") or {}
                has_hash = bool(raw.get("entry_hash") or event.get("entry_hash"))
                if has_hash:
                    payload = raw if raw.get("entry_hash") else event
                    if not self._audit.verify_entry_data(payload):
                        tampered.append(event.get("audit_id") or event["checkpoint"])
                elif must_hash:
                    unhashed.append(event.get("audit_id") or event["checkpoint"] or "unknown")

            found = len(expected_checkpoints) - len(missing)
            score = found / len(expected_checkpoints)
            passed = not missing and not tampered and not unhashed
            result = ValidationResult(
                passed=passed,
                score=_clamp(score),
                evaluator_name=self.name,
                metric="audit_trail_completeness",
                threshold=1.0,
                details={
                    "agent_run_id": agent_run_id,
                    "expected_checkpoints": list(expected_checkpoints),
                    "present_checkpoints": present,
                    "missing_checkpoints": missing,
                    "order_ok": order_ok,
                    "tampered_ids": tampered,
                    "unhashed_ids": unhashed,
                    "event_count": len(normalized),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "audit_trail_completeness", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_human_override(
        self,
        *,
        agent_run_id: str,
        override_event: OverrideEvent,
        events: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm an override was logged and applied downstream.

        Passes when (1) a matching override record exists for the checkpoint
        and (2) no later event on the same run repeats `original_action`
        without override metadata. If later events exist, at least one should
        carry `override_action`.

        Args:
            agent_run_id: Run id to filter events.
            override_event: The override that must appear in the trail.
            events: In-memory trail. Loaded from AuditLogger when omitted.

        Returns:
            ValidationResult. score is 1.0 when recorded and applied.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            run_id = agent_run_id or override_event.run_id
            self._require_text(run_id, "agent_run_id", "human_override")
            normalized = await self._load_events(events, run_id)
            recorded = False
            recorded_event: dict[str, Any] | None = None
            for event in normalized:
                if _is_override_record(event, override_event):
                    recorded = True
                    recorded_event = event
                    break

            later = []
            if recorded_event is not None:
                idx = normalized.index(recorded_event)
                later = normalized[idx + 1 :]
            else:
                later = list(normalized)

            reverted = [
                event
                for event in later
                if event["action"] == override_event.original_action
                and not event["overridden"]
                and event["agent"] == override_event.agent
            ]
            applied = any(
                event["action"] == override_event.override_action
                or (
                    event["overridden"]
                    and event.get("override_action") == override_event.override_action
                )
                for event in later
            )
            # No downstream events: recording the override is sufficient.
            if not later:
                applied = recorded

            passed = recorded and applied and not reverted
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="human_override",
                threshold=1.0,
                details={
                    "agent_run_id": run_id,
                    "checkpoint": override_event.checkpoint,
                    "agent": override_event.agent,
                    "original_action": override_event.original_action,
                    "override_action": override_event.override_action,
                    "override_by": override_event.override_by,
                    "recorded": recorded,
                    "applied": applied,
                    "reverted_to_original": [e.get("checkpoint") for e in reverted],
                    "downstream_event_count": len(later),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "human_override", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_structural_separation(
        self,
        *,
        agent_a: str,
        agent_b: str,
        forbidden_actions: list[str],
        events: list[dict[str, Any]] | None = None,
        agent_run_id: str = "",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm agent_a never performed actions reserved for agent_b.

        Zero-tolerance: any forbidden action by agent_a fails the check.

        Args:
            agent_a: Agent that must not perform forbidden_actions.
            agent_b: Agent that owns those actions (details only).
            forbidden_actions: Action names reserved for agent_b.
            events: In-memory trail. Loaded from AuditLogger when omitted.
            agent_run_id: Optional run id filter.

        Returns:
            ValidationResult. score is 1.0 on zero violations, else 0.0.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(agent_a, "agent_a", "structural_separation")
            self._require_text(agent_b, "agent_b", "structural_separation")
            if not forbidden_actions:
                raise GovernanceValidatorError(
                    "forbidden_actions must be a non-empty list",
                    evaluator_name=self.name,
                    metric="structural_separation",
                )
            forbidden = {action for action in forbidden_actions}
            normalized = await self._load_events(events, agent_run_id or None)
            violations = [
                {
                    "checkpoint": event["checkpoint"],
                    "action": event["action"],
                    "agent": event["agent"],
                }
                for event in normalized
                if event["agent"] == agent_a and event["action"] in forbidden
            ]
            passed = not violations
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="structural_separation",
                threshold=1.0,
                details={
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "forbidden_actions": list(forbidden_actions),
                    "violations": violations,
                    "event_count": len(normalized),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "structural_separation", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_tamper_evidence(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        agent_run_id: str = "",
        **kwargs: Any,
    ) -> ValidationResult:
        """Verify SHA-256 hashes on every hashed entry in the trail.

        Args:
            events: In-memory trail. Loaded from AuditLogger when omitted.
            agent_run_id: Optional run id filter.

        Returns:
            ValidationResult. score is valid/total hashed entries. passed
            only when every hashed entry verifies and at least one exists.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            source = events
            if source is None:
                source = await self._audit.read_entries()
            if agent_run_id:
                source = [
                    item
                    for item in source
                    if (rid := _normalize_event(item)["run_id"]) in ("", agent_run_id)
                ]
            hashed = [item for item in source if item.get("entry_hash")]
            if not hashed:
                raise GovernanceValidatorError(
                    "tamper_evidence requires hashed audit entries",
                    evaluator_name=self.name,
                    metric="tamper_evidence",
                )
            valid = [item for item in hashed if self._audit.verify_entry_data(item)]
            invalid_ids = [
                str(item.get("audit_id") or "")
                for item in hashed
                if not self._audit.verify_entry_data(item)
            ]
            score = len(valid) / len(hashed)
            result = ValidationResult(
                passed=len(valid) == len(hashed),
                score=_clamp(score),
                evaluator_name=self.name,
                metric="tamper_evidence",
                threshold=1.0,
                details={
                    "hashed_count": len(hashed),
                    "valid_count": len(valid),
                    "invalid_ids": [i for i in invalid_ids if i],
                    "agent_run_id": agent_run_id,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "tamper_evidence", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        agent_run_id: str,
        expected_checkpoints: list[str],
        events: list[dict[str, Any]] | None = None,
        override_event: OverrideEvent | None = None,
        agent_a: str = "",
        agent_b: str = "",
        forbidden_actions: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Run completeness plus any optional override/separation checks.

        Args:
            agent_run_id: Run id shared by all checks.
            expected_checkpoints: Required checkpoints.
            events: In-memory trail. Loaded from AuditLogger when omitted.
            override_event: If set, also run human_override.
            agent_a / agent_b / forbidden_actions: If set, also run
                structural_separation.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results = [
            await self.validate_audit_trail_completeness(
                agent_run_id=agent_run_id,
                expected_checkpoints=expected_checkpoints,
                events=events,
            )
        ]
        if override_event is not None:
            results.append(
                await self.validate_human_override(
                    agent_run_id=agent_run_id,
                    override_event=override_event,
                    events=events,
                )
            )
        if agent_a and agent_b and forbidden_actions:
            results.append(
                await self.validate_structural_separation(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    forbidden_actions=forbidden_actions,
                    events=events,
                    agent_run_id=agent_run_id,
                )
            )
        return results

    async def _load_events(
        self,
        events: list[dict[str, Any]] | None,
        run_id: str | None,
    ) -> list[dict[str, Any]]:
        """Load and normalize events, optionally filtered by run id.

        Args:
            events: Caller-supplied trail, or None to read JSONL logs.
            run_id: If non-empty, keep events whose run_id matches.

        Returns:
            Normalized event dicts.
        """
        source = events
        if source is None:
            source = await self._audit.read_entries()
        normalized = [_normalize_event(item) for item in source]
        if run_id:
            normalized = [
                event
                for event in normalized
                if not event["run_id"] or event["run_id"] == run_id
            ]
        return normalized

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise GovernanceValidatorError if `value` is empty or whitespace."""
        if not isinstance(value, str) or not value.strip():
            raise GovernanceValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten an in-memory event or AuditLogger JSONL row.

    Args:
        raw: Event dict. Checkpoint/agent/action may live at the top level
            or under `details`.

    Returns:
        Normalized event with checkpoint, agent, action, run_id, and raw.
    """
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    return {
        "checkpoint": str(
            raw.get("checkpoint") or details.get("checkpoint") or raw.get("metric") or ""
        ),
        "agent": str(raw.get("agent") or details.get("agent") or raw.get("evaluator") or ""),
        "action": str(raw.get("action") or details.get("action") or ""),
        "run_id": str(
            raw.get("run_id")
            or raw.get("agent_run_id")
            or details.get("run_id")
            or details.get("agent_run_id")
            or ""
        ),
        "audit_id": str(raw.get("audit_id") or ""),
        "entry_hash": str(raw.get("entry_hash") or ""),
        "timestamp": str(raw.get("timestamp") or details.get("timestamp") or ""),
        "overridden": bool(raw.get("overridden") or details.get("overridden")),
        "override_by": raw.get("override_by") or details.get("override_by"),
        "override_action": raw.get("override_action") or details.get("override_action"),
        "original_action": raw.get("original_action") or details.get("original_action"),
        "raw": raw,
    }


def _coerce_override(value: OverrideEvent | dict[str, Any]) -> OverrideEvent:
    """Build an OverrideEvent from a dataclass or dict.

    Args:
        value: OverrideEvent or mapping of the same fields.

    Returns:
        OverrideEvent instance.
    """
    if isinstance(value, OverrideEvent):
        return value
    return OverrideEvent(
        run_id=str(value.get("run_id") or value.get("agent_run_id") or ""),
        checkpoint=str(value.get("checkpoint") or ""),
        agent=str(value.get("agent") or ""),
        original_action=str(value.get("original_action") or ""),
        override_action=str(value.get("override_action") or ""),
        override_by=str(value.get("override_by") or ""),
        timestamp=str(value.get("timestamp") or ""),
    )


def _is_override_record(event: dict[str, Any], override: OverrideEvent) -> bool:
    """True if `event` is the logged override matching `override`.

    Args:
        event: Normalized trail event.
        override: Expected override.

    Returns:
        True when checkpoint/run match and override metadata is present.
    """
    if override.checkpoint and event["checkpoint"] != override.checkpoint:
        return False
    if override.run_id and event["run_id"] and event["run_id"] != override.run_id:
        return False
    if event["overridden"]:
        return True
    if event.get("override_by") and str(event.get("override_by")) == override.override_by:
        return True
    if event["action"] == override.override_action and event.get("override_by"):
        return True
    return False


def _checkpoints_in_order(present: list[str], expected: list[str]) -> bool:
    """True if expected checkpoints appear in `present` in the same order.

    Args:
        present: Checkpoints observed in the trail (may include extras).
        expected: Required checkpoint sequence.

    Returns:
        True if expected is an ordered subsequence of present.
    """
    iterator = iter(present)
    for checkpoint in expected:
        for observed in iterator:
            if observed == checkpoint:
                break
        else:
            return False
    return True


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
