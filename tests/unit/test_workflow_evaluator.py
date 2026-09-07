"""Unit tests for WorkflowEvaluator (trace scoring, not a runtime)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError
from agentproof.evaluators.workflow import (
    AgentSpan,
    ToolCall,
    WorkflowEvaluator,
    WorkflowTrace,
)


def _trace() -> WorkflowTrace:
    return WorkflowTrace(
        run_id="run-1",
        spans=[
            AgentSpan(
                name="researcher",
                kind="subagent",
                tools=[ToolCall(name="web.search", server="brave")],
            ),
            AgentSpan(name="reviewer", kind="subagent"),
            AgentSpan(name="pdf", kind="skill"),
            AgentSpan(name="nightly-eval", kind="background", completed=True),
            AgentSpan(name="approve", kind="pr_review"),
        ],
    )


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(mock_audit) -> WorkflowEvaluator:
    return WorkflowEvaluator(AgentProofConfig(), mock_audit)


def test_name(evaluator):
    assert evaluator.name == "WorkflowEvaluator"


async def test_subagents_pass(evaluator):
    result = await evaluator.validate_subagents(
        trace=_trace(), expected_agents=["researcher", "reviewer"]
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_subagents_missing(evaluator):
    result = await evaluator.validate_subagents(
        trace=_trace(), expected_agents=["researcher", "writer"]
    )
    assert result.passed is False
    assert result.details["missing"] == ["writer"]
    assert abs(result.score - 0.5) < 1e-9


async def test_skills_pass(evaluator):
    result = await evaluator.validate_skills(trace=_trace(), expected_skills=["pdf"])
    assert result.passed is True


async def test_mcp_allowlist_pass(evaluator):
    result = await evaluator.validate_mcp(
        trace=_trace(), allowed_tools=["web.search", "github.get_pr"]
    )
    assert result.passed is True


async def test_mcp_denies_unknown_tool(evaluator):
    result = await evaluator.validate_mcp(trace=_trace(), allowed_tools=["github.get_pr"])
    assert result.passed is False
    assert result.details["denied"] == ["web.search"]


async def test_background_incomplete_fails(evaluator):
    trace = WorkflowTrace(
        run_id="run-2",
        spans=[AgentSpan(name="nightly-eval", kind="background", completed=False)],
    )
    result = await evaluator.validate_background(
        trace=trace, expected_jobs=["nightly-eval"]
    )
    assert result.passed is False
    assert result.details["incomplete"] == ["nightly-eval"]


async def test_pr_review_gate(evaluator):
    result = await evaluator.validate_pr_review(
        trace=_trace(), required_gates=["approve"]
    )
    assert result.passed is True


async def test_evaluate_accepts_dict_trace(evaluator):
    result = await evaluator.evaluate(
        metric="subagents",
        trace={"run_id": "r", "spans": [{"name": "a", "kind": "subagent"}]},
        expected_agents=["a"],
    )
    assert result.passed is True


async def test_unknown_metric(evaluator, mock_audit):
    result = await evaluator.evaluate(metric="swarm", trace=_trace())
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


async def test_audit_error_propagates(evaluator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError):
        await evaluator.validate_skills(trace=_trace(), expected_skills=["pdf"])
