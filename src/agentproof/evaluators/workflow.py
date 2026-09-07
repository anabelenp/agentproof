"""WorkflowEvaluator — scores traces of agentic runs, not a runtime.

AgentProof does not host subagents, skills, MCP servers, or PR bots. This
evaluator checks a recorded trace: expected subagents ran, skills fired,
MCP tools stayed on the allowlist, background work completed, and PR-review
gates were present.

Usage:
    evaluator = WorkflowEvaluator(config, audit_logger)
    result = await evaluator.validate_subagents(
        trace=trace,
        expected_agents=["researcher", "reviewer"],
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, WorkflowEvaluatorError

SUPPORTED_METRICS = (
    "subagents",
    "skills",
    "mcp",
    "background",
    "pr_review",
)


@dataclass
class ToolCall:
    """One tool or MCP invocation in a workflow trace.

    Args:
        name: Tool name (`search`, `github.create_pr`).
        server: Optional MCP server id.
        arguments: Serialized arguments (no secrets).
        error: Tool error string, if the call failed.
        completed: True when the call returned.
    """

    name: str
    server: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed: bool = True


@dataclass
class AgentSpan:
    """One agent / skill / background unit of work.

    Args:
        name: Subagent, skill, or job name.
        kind: subagent, skill, mcp, background, or pr_review.
        tools: Tool calls made during this span.
        completed: True when the span finished (not still running).
        parent: Optional parent agent name.
    """

    name: str
    kind: str
    tools: list[ToolCall] = field(default_factory=list)
    completed: bool = True
    parent: str = ""


@dataclass
class WorkflowTrace:
    """Recorded agentic run used as evaluation input.

    Args:
        run_id: Id of the agent run under test.
        spans: Ordered spans (subagents, skills, background jobs, reviews).
    """

    run_id: str
    spans: list[AgentSpan] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[AgentSpan]:
        """Return spans whose kind matches `kind` (case-insensitive)."""
        key = kind.lower()
        return [span for span in self.spans if span.kind.lower() == key]

    def tool_names(self) -> list[str]:
        """Flatten tool names across all spans."""
        names: list[str] = []
        for span in self.spans:
            for tool in span.tools:
                names.append(tool.name)
        return names


def coerce_trace(value: WorkflowTrace | dict[str, Any]) -> WorkflowTrace:
    """Build a WorkflowTrace from a dataclass or mapping.

    Args:
        value: WorkflowTrace or dict with `run_id` and `spans`.

    Returns:
        WorkflowTrace instance.
    """
    if isinstance(value, WorkflowTrace):
        return value
    spans: list[AgentSpan] = []
    for raw in value.get("spans") or []:
        if isinstance(raw, AgentSpan):
            spans.append(raw)
            continue
        tools = []
        for tool in raw.get("tools") or []:
            if isinstance(tool, ToolCall):
                tools.append(tool)
            else:
                tools.append(
                    ToolCall(
                        name=str(tool.get("name") or ""),
                        server=str(tool.get("server") or ""),
                        arguments=dict(tool.get("arguments") or {}),
                        error=tool.get("error"),
                        completed=bool(tool.get("completed", True)),
                    )
                )
        spans.append(
            AgentSpan(
                name=str(raw.get("name") or ""),
                kind=str(raw.get("kind") or "subagent"),
                tools=tools,
                completed=bool(raw.get("completed", True)),
                parent=str(raw.get("parent") or ""),
            )
        )
    return WorkflowTrace(run_id=str(value.get("run_id") or ""), spans=spans)


class WorkflowEvaluator(BaseEvaluator):
    """Scores a recorded agentic workflow against expected structure.

    Args:
        config: AgentProofConfig.
        audit_logger: AuditLogger that receives every result.
    """

    def __init__(self, config: AgentProofConfig, audit_logger: AuditLogger) -> None:
        super().__init__(config, audit_logger)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "WorkflowEvaluator"

    async def evaluate(
        self,
        *,
        metric: str = "subagents",
        trace: WorkflowTrace | dict[str, Any] | None = None,
        expected_agents: list[str] | None = None,
        expected_skills: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        expected_jobs: list[str] | None = None,
        required_gates: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a workflow check.

        Args:
            metric: subagents (default), skills, mcp, background, pr_review.
            trace: Recorded WorkflowTrace or dict.
            expected_agents: Subagent names that must appear.
            expected_skills: Skill names that must appear.
            allowed_tools: MCP / tool allowlist.
            expected_jobs: Background job names that must complete.
            required_gates: PR-review checkpoint names.

        Returns:
            ValidationResult for the requested metric.
        """
        parsed = coerce_trace(trace or {})
        key = metric.lower().strip()
        if key == "subagents":
            return await self.validate_subagents(
                trace=parsed, expected_agents=expected_agents or []
            )
        if key == "skills":
            return await self.validate_skills(
                trace=parsed, expected_skills=expected_skills or []
            )
        if key == "mcp":
            return await self.validate_mcp(
                trace=parsed, allowed_tools=allowed_tools or []
            )
        if key == "background":
            return await self.validate_background(
                trace=parsed, expected_jobs=expected_jobs or []
            )
        if key in {"pr_review", "pr-review"}:
            return await self.validate_pr_review(
                trace=parsed, required_gates=required_gates or []
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

    async def validate_subagents(
        self,
        *,
        trace: WorkflowTrace,
        expected_agents: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm expected subagents ran and completed.

        Args:
            trace: Recorded run.
            expected_agents: Names that must appear as kind=subagent.

        Returns:
            ValidationResult. score is found/expected.
        """
        return await self._coverage(
            metric="subagents",
            trace=trace,
            kind="subagent",
            expected=expected_agents,
            require_complete=True,
        )

    async def validate_skills(
        self,
        *,
        trace: WorkflowTrace,
        expected_skills: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm expected skills were invoked.

        Args:
            trace: Recorded run.
            expected_skills: Skill names that must appear as kind=skill.

        Returns:
            ValidationResult. score is found/expected.
        """
        return await self._coverage(
            metric="skills",
            trace=trace,
            kind="skill",
            expected=expected_skills,
            require_complete=True,
        )

    async def validate_mcp(
        self,
        *,
        trace: WorkflowTrace,
        allowed_tools: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm every tool call is on the MCP / tool allowlist.

        Args:
            trace: Recorded run.
            allowed_tools: Permitted tool names.

        Returns:
            ValidationResult. score is allowed-calls / total-calls.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not allowed_tools:
                raise WorkflowEvaluatorError(
                    "mcp requires a non-empty allowed_tools list",
                    evaluator_name=self.name,
                    metric="mcp",
                )
            permitted = {name.strip() for name in allowed_tools if name.strip()}
            called = trace.tool_names()
            denied = [name for name in called if name not in permitted]
            total = len(called)
            score = ((total - len(denied)) / total) if total else 1.0
            result = ValidationResult(
                passed=not denied,
                score=score if 0.0 <= score <= 1.0 else 0.0,
                evaluator_name=self.name,
                metric="mcp",
                threshold=1.0,
                details={
                    "run_id": trace.run_id,
                    "called": called,
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
            result = self._error_result(audit_id, start, "mcp", exc, threshold=1.0)

        await self._write_audit(result)
        return result

    async def validate_background(
        self,
        *,
        trace: WorkflowTrace,
        expected_jobs: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm background jobs were started and completed.

        Args:
            trace: Recorded run.
            expected_jobs: kind=background span names.

        Returns:
            ValidationResult. Incomplete jobs fail even if named.
        """
        return await self._coverage(
            metric="background",
            trace=trace,
            kind="background",
            expected=expected_jobs,
            require_complete=True,
        )

    async def validate_pr_review(
        self,
        *,
        trace: WorkflowTrace,
        required_gates: list[str],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm PR-review checkpoints ran.

        Args:
            trace: Recorded run.
            required_gates: kind=pr_review span names (e.g. `lint`, `approve`).

        Returns:
            ValidationResult. score is found/expected.
        """
        return await self._coverage(
            metric="pr_review",
            trace=trace,
            kind="pr_review",
            expected=required_gates,
            require_complete=True,
        )

    async def _coverage(
        self,
        *,
        metric: str,
        trace: WorkflowTrace,
        kind: str,
        expected: list[str],
        require_complete: bool,
    ) -> ValidationResult:
        """Score expected-name coverage for spans of a given kind.

        Args:
            metric: Result metric name.
            trace: Recorded run.
            kind: Span kind filter.
            expected: Names that must appear.
            require_complete: If True, incomplete matching spans count as missing.

        Returns:
            ValidationResult.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not expected:
                raise WorkflowEvaluatorError(
                    f"{metric} requires a non-empty expected list",
                    evaluator_name=self.name,
                    metric=metric,
                )
            present = trace.of_kind(kind)
            names = {span.name for span in present}
            completed = {span.name for span in present if span.completed}
            missing = [name for name in expected if name not in names]
            incomplete = [
                name
                for name in expected
                if name in names and require_complete and name not in completed
            ]
            found = len(expected) - len(missing)
            score = found / len(expected)
            passed = not missing and not incomplete
            result = ValidationResult(
                passed=passed,
                score=score if 0.0 <= score <= 1.0 else 0.0,
                evaluator_name=self.name,
                metric=metric,
                threshold=1.0,
                details={
                    "run_id": trace.run_id,
                    "kind": kind,
                    "expected": list(expected),
                    "present": sorted(names),
                    "missing": missing,
                    "incomplete": incomplete,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(audit_id, start, metric, exc, threshold=1.0)

        await self._write_audit(result)
        return result
