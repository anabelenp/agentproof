"""Unit tests for GovernanceValidator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentproof.core.audit import AuditEntry, AuditLogger, new_audit_id
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError
from agentproof.core.runner import TestRunner
from agentproof.validators.governance import GovernanceValidator, OverrideEvent

RUN_ID = "run-underwriting-1"


def _event(
    checkpoint: str,
    *,
    agent: str = "underwriting_agent",
    action: str = "draft",
    run_id: str = RUN_ID,
    **extra,
) -> dict:
    row = {
        "checkpoint": checkpoint,
        "agent": agent,
        "action": action,
        "run_id": run_id,
    }
    row.update(extra)
    return row


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    audit.read_entries = AsyncMock(return_value=[])
    audit.verify_entry_data = MagicMock(return_value=True)
    return audit


@pytest.fixture
def validator(mock_audit) -> GovernanceValidator:
    return GovernanceValidator(AgentProofConfig(), mock_audit)


# ── identity ──────────────────────────────────────────────────────────────────


def test_name(validator):
    assert validator.name == "GovernanceValidator"


async def test_unknown_metric(validator, mock_audit):
    result = await validator.evaluate(metric="sox")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── completeness ──────────────────────────────────────────────────────────────


async def test_completeness_pass(validator):
    events = [
        _event("retrieve", action="search"),
        _event("draft", action="write"),
        _event("human_review", action="approve"),
        _event("issue", action="publish"),
    ]
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "draft", "human_review", "issue"],
        events=events,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["missing_checkpoints"] == []
    assert result.details["order_ok"] is True
    assert result.error is None


async def test_completeness_missing_checkpoint(validator):
    events = [_event("retrieve"), _event("draft")]
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "draft", "human_review"],
        events=events,
    )
    assert result.passed is False
    assert abs(result.score - 2 / 3) < 1e-9
    assert result.details["missing_checkpoints"] == ["human_review"]
    assert result.error is None


async def test_completeness_empty_events_is_quality_failure(validator):
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "draft"],
        events=[],
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.error is None


async def test_completeness_detects_out_of_order(validator):
    events = [_event("draft"), _event("retrieve")]
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "draft"],
        events=events,
    )
    assert result.details["order_ok"] is False
    # still complete (both present) — order is reported, not a hard fail
    assert result.passed is True


async def test_completeness_requires_run_id(validator, mock_audit):
    result = await validator.validate_audit_trail_completeness(
        agent_run_id="  ",
        expected_checkpoints=["retrieve"],
        events=[],
    )
    assert result.error is not None
    assert "agent_run_id" in result.error
    mock_audit.log.assert_awaited_once()


async def test_completeness_requires_checkpoints(validator):
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=[],
        events=[],
    )
    assert result.error is not None
    assert "expected_checkpoints" in result.error


async def test_completeness_reads_logger_and_requires_hashes(validator, mock_audit):
    mock_audit.read_entries = AsyncMock(
        return_value=[
            {
                "audit_id": "a1",
                "entry_hash": "sha256:abc",
                "details": {
                    "run_id": RUN_ID,
                    "checkpoint": "retrieve",
                    "agent": "underwriting_agent",
                    "action": "search",
                },
            }
        ]
    )
    mock_audit.verify_entry_data = MagicMock(return_value=False)
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve"],
    )
    assert result.passed is False
    assert result.details["tampered_ids"] == ["a1"]


async def test_completeness_writes_audit(validator, mock_audit):
    result = await validator.validate_audit_trail_completeness(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve"],
        events=[_event("retrieve")],
    )
    mock_audit.log.assert_awaited_once()
    assert mock_audit.log.call_args[0][0].audit_id == result.audit_id


# ── human override ────────────────────────────────────────────────────────────


def _override() -> OverrideEvent:
    return OverrideEvent(
        run_id=RUN_ID,
        checkpoint="human_review",
        agent="underwriting_agent",
        original_action="approve",
        override_action="deny",
        override_by="senior_adjuster",
    )


async def test_override_recorded_and_applied(validator):
    events = [
        _event("draft", action="approve"),
        _event(
            "human_review",
            action="deny",
            overridden=True,
            override_by="senior_adjuster",
            override_action="deny",
            original_action="approve",
        ),
        _event("issue", action="deny"),
    ]
    result = await validator.validate_human_override(
        agent_run_id=RUN_ID, override_event=_override(), events=events
    )
    assert result.passed is True
    assert result.details["recorded"] is True
    assert result.details["applied"] is True


async def test_override_fails_when_not_logged(validator):
    events = [_event("draft", action="approve"), _event("issue", action="approve")]
    result = await validator.validate_human_override(
        agent_run_id=RUN_ID, override_event=_override(), events=events
    )
    assert result.passed is False
    assert result.details["recorded"] is False


async def test_override_fails_when_downstream_reverts(validator):
    events = [
        _event(
            "human_review",
            action="deny",
            overridden=True,
            override_by="senior_adjuster",
            override_action="deny",
        ),
        _event("issue", action="approve"),
    ]
    result = await validator.validate_human_override(
        agent_run_id=RUN_ID, override_event=_override(), events=events
    )
    assert result.passed is False
    assert result.details["reverted_to_original"] == ["issue"]


async def test_override_logged_without_downstream_passes(validator):
    events = [
        _event(
            "human_review",
            action="deny",
            overridden=True,
            override_by="senior_adjuster",
        )
    ]
    result = await validator.validate_human_override(
        agent_run_id=RUN_ID, override_event=_override(), events=events
    )
    assert result.passed is True
    assert result.details["downstream_event_count"] == 0


async def test_evaluate_override_requires_event(validator, mock_audit):
    result = await validator.evaluate(metric="human_override", agent_run_id=RUN_ID)
    assert result.error is not None
    assert "override_event" in result.error


# ── structural separation ─────────────────────────────────────────────────────


async def test_separation_pass(validator):
    events = [
        _event("draft", agent="writer", action="draft_finding"),
        _event("approve", agent="reviewer", action="approve_finding"),
    ]
    result = await validator.validate_structural_separation(
        agent_a="writer",
        agent_b="reviewer",
        forbidden_actions=["approve_finding"],
        events=events,
    )
    assert result.passed is True
    assert result.details["violations"] == []


async def test_separation_detects_bypass(validator):
    events = [
        _event("draft", agent="writer", action="draft_finding"),
        _event("sneak", agent="writer", action="approve_finding"),
    ]
    result = await validator.validate_structural_separation(
        agent_a="writer",
        agent_b="reviewer",
        forbidden_actions=["approve_finding"],
        events=events,
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["violations"][0]["action"] == "approve_finding"


async def test_separation_requires_forbidden_actions(validator):
    result = await validator.validate_structural_separation(
        agent_a="writer",
        agent_b="reviewer",
        forbidden_actions=[],
        events=[],
    )
    assert result.error is not None


# ── tamper evidence ───────────────────────────────────────────────────────────


async def test_tamper_evidence_pass_from_logger(tmp_path):
    logger = AuditLogger(log_dir=tmp_path / "audit_logs")
    entry = AuditEntry(
        audit_id=new_audit_id(),
        evaluator="agent_runtime",
        metric="checkpoint",
        score=1.0,
        threshold=1.0,
        passed=True,
        model="",
        latency_ms=1.0,
        details={"run_id": RUN_ID, "checkpoint": "retrieve"},
    )
    await logger.log(entry)
    validator = GovernanceValidator(AgentProofConfig(), logger)
    result = await validator.validate_tamper_evidence(agent_run_id=RUN_ID)
    assert result.passed is True
    assert result.details["hashed_count"] == 1


async def test_tamper_evidence_detects_mutation(tmp_path):
    logger = AuditLogger(log_dir=tmp_path / "audit_logs")
    entry = AuditEntry(
        audit_id=new_audit_id(),
        evaluator="agent_runtime",
        metric="checkpoint",
        score=1.0,
        threshold=1.0,
        passed=True,
        model="",
        latency_ms=1.0,
        details={"run_id": RUN_ID, "checkpoint": "retrieve"},
    )
    await logger.log(entry)
    path = logger._today_log_path()
    import json

    data = json.loads(path.read_text().strip())
    data["score"] = 0.0
    path.write_text(json.dumps(data) + "\n")

    validator = GovernanceValidator(AgentProofConfig(), logger)
    result = await validator.validate_tamper_evidence()
    assert result.passed is False
    assert result.details["invalid_ids"] == [entry.audit_id]


async def test_tamper_evidence_errors_without_hashes(validator):
    result = await validator.validate_tamper_evidence(events=[{"checkpoint": "x"}])
    assert result.error is not None
    assert "hashed" in result.error


# ── evaluate_all / runner / AuditError ────────────────────────────────────────


async def test_evaluate_all_runs_optional_checks(validator):
    events = [
        _event("retrieve", agent="writer", action="search"),
        _event(
            "human_review",
            agent="writer",
            action="deny",
            overridden=True,
            override_by="senior_adjuster",
        ),
    ]
    results = await validator.evaluate_all(
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "human_review"],
        events=events,
        override_event=_override(),
        agent_a="writer",
        agent_b="reviewer",
        forbidden_actions=["approve_finding"],
    )
    assert [r.metric for r in results] == [
        "audit_trail_completeness",
        "human_override",
        "structural_separation",
    ]


async def test_audit_error_propagates(validator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError):
        await validator.validate_audit_trail_completeness(
            agent_run_id=RUN_ID,
            expected_checkpoints=["retrieve"],
            events=[_event("retrieve")],
        )


async def test_runner_registers_governance(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    validator = GovernanceValidator(config, runner.audit_logger)
    events = [_event("retrieve"), _event("draft")]
    runner.register(
        validator,
        agent_run_id=RUN_ID,
        expected_checkpoints=["retrieve", "draft"],
        events=events,
        metric="audit_trail_completeness",
    )
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1
