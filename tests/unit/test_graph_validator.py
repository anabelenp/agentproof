"""Unit tests for Neo4jIntegration and GraphValidator. All Bolt calls mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError
from agentproof.core.runner import TestRunner
from agentproof.integrations.neo4j import (
    Neo4jIntegration,
    extract_node_id,
    quote_ident,
    record_to_dict,
)
from agentproof.validators.graph import (
    GraphValidator,
    events_in_order,
    group_sources,
    normalize_direction,
)
from neo4j.exceptions import ServiceUnavailable


def _async_cm(value=None):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _async_iter(items):
    async def _gen():
        for item in items:
            yield item

    return _gen()


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = AsyncMock()
    result = MagicMock()
    result.__aiter__ = lambda self: _async_iter([])
    result.consume = AsyncMock(return_value=MagicMock(counters=MagicMock(nodes_created=0, relationships_created=0)))
    session.run = AsyncMock(return_value=result)
    driver.session = MagicMock(return_value=_async_cm(session))
    driver.close = AsyncMock()
    return driver, session, result


@pytest.fixture
def integration(mock_driver) -> Neo4jIntegration:
    driver, _, _ = mock_driver
    return Neo4jIntegration(AgentProofConfig(), driver=driver)


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def validator(mock_audit, integration) -> GraphValidator:
    return GraphValidator(AgentProofConfig(), mock_audit, integration=integration)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_quote_ident():
    assert quote_ident("Entity") == "`Entity`"


def test_quote_ident_rejects_injection():
    with pytest.raises(IntegrationError, match="invalid Cypher identifier"):
        quote_ident("Entity} DELETE n //")


def test_record_to_dict_from_mapping_and_data():
    assert record_to_dict({"id": "n1"}) == {"id": "n1"}
    assert record_to_dict(None) == {}

    class Rec:
        def data(self):
            return {"id": "n2"}

    assert record_to_dict(Rec()) == {"id": "n2"}


def test_extract_node_id_fallbacks():
    assert extract_node_id({"id": "a"}) == "a"
    assert extract_node_id({"node_id": "b"}) == "b"
    assert extract_node_id({"n": {"id": "c"}}) == "c"
    assert extract_node_id({}) == ""


def test_normalize_direction_aliases():
    assert normalize_direction("->") == "outgoing"
    assert normalize_direction("INCOMING") == "incoming"
    assert normalize_direction("<->") == "both"


def test_group_sources_by_entity_key():
    grouped = group_sources(
        [
            {"entity_key": "acme", "email": "a@acme.com"},
            {"entity_key": "acme", "email": "b@acme.com"},
            {"entity_key": "beta", "email": "c@beta.com"},
        ]
    )
    assert len(grouped["acme"]) == 2
    assert len(grouped["beta"]) == 1


def test_events_in_order():
    assert events_in_order(["a", "x", "b", "c"], ["a", "b", "c"]) is True
    assert events_in_order(["b", "a", "c"], ["a", "b", "c"]) is False


# ── Neo4jIntegration ──────────────────────────────────────────────────────────


def test_init_calls_async_driver():
    with patch("agentproof.integrations.neo4j.AsyncGraphDatabase.driver") as mocked:
        mocked.return_value = MagicMock()
        Neo4jIntegration(AgentProofConfig(neo4j_url="bolt://graph:7687", neo4j_user="neo4j"))
    mocked.assert_called_once()
    assert mocked.call_args.args[0] == "bolt://graph:7687"
    assert mocked.call_args.kwargs["auth"] == ("neo4j", "")


async def test_query_returns_dicts(integration, mock_driver):
    _, session, result = mock_driver
    rec = MagicMock()
    rec.data = lambda: {"id": "n1"}
    result.__aiter__ = lambda self: _async_iter([rec])
    rows = await integration.query("MATCH (n) RETURN n.id AS id")
    assert rows == [{"id": "n1"}]
    session.run.assert_awaited()


async def test_query_empty_cypher_raises(integration):
    with pytest.raises(IntegrationError, match="cypher"):
        await integration.query("  ")


async def test_query_retries_service_unavailable(integration, mock_driver):
    _, session, result = mock_driver
    rec = MagicMock()
    rec.data = lambda: {"id": "ok"}
    session.run = AsyncMock(
        side_effect=[ServiceUnavailable("down"), result]
    )
    result.__aiter__ = lambda self: _async_iter([rec])
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        rows = await integration.query("MATCH (n) RETURN n")
    assert rows == [{"id": "ok"}]
    assert session.run.await_count == 2


async def test_query_exhausted_retries_become_integration_error(integration, mock_driver):
    _, session, _ = mock_driver
    session.run = AsyncMock(side_effect=ServiceUnavailable("down"))
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.query("MATCH (n) RETURN n")


async def test_close_delegates(integration, mock_driver):
    driver, _, _ = mock_driver
    await integration.close()
    driver.close.assert_awaited_once()


# ── GraphValidator identity ───────────────────────────────────────────────────


def test_name(validator):
    assert validator.name == "GraphValidator"


async def test_unknown_metric(validator, mock_audit):
    result = await validator.evaluate(metric="pagerank")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── entity_resolution ─────────────────────────────────────────────────────────


async def test_entity_resolution_pass(validator):
    sources = [
        {"entity_key": "acme", "email": "a@acme.com", "source": "crm"},
        {"entity_key": "acme", "email": "acme@mail.com", "source": "email"},
    ]
    result = await validator.validate_entity_resolution(
        entity_sources=sources,
        expected_node_count=1,
        resolved_nodes=[{"entity_key": "acme", "id": "node-1"}],
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.metric == "entity_resolution"
    assert result.threshold == 0.95


async def test_entity_resolution_split_nodes(validator):
    sources = [
        {"entity_key": "acme", "email": "a@acme.com"},
        {"entity_key": "acme", "email": "b@acme.com"},
    ]
    result = await validator.validate_entity_resolution(
        entity_sources=sources,
        expected_node_count=1,
        resolved_nodes=[
            {"entity_key": "acme", "id": "n1"},
            {"entity_key": "acme", "id": "n2"},
        ],
    )
    assert result.passed is False
    assert result.details["over_merged_entities"] == ["acme"]
    assert result.error is None


async def test_entity_resolution_missing_entity(validator):
    result = await validator.validate_entity_resolution(
        entity_sources=[{"entity_key": "ghost", "email": "x@y.z"}],
        resolved_nodes=[],
    )
    assert result.passed is False
    assert result.details["missing_entities"] == ["ghost"]
    assert result.score == 0.0


async def test_entity_resolution_empty_sources_is_error(validator, mock_audit):
    result = await validator.validate_entity_resolution(entity_sources=[])
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


async def test_entity_resolution_queries_neo4j(validator, mock_driver):
    _, session, result_obj = mock_driver
    rec = MagicMock()
    rec.data = lambda: {"id": "node-9"}
    result_obj.__aiter__ = lambda self: _async_iter([rec])
    result = await validator.validate_entity_resolution(
        entity_sources=[{"entity_key": "acme", "email": "a@acme.com"}],
        expected_node_count=1,
    )
    assert result.passed is True
    session.run.assert_awaited()


# ── relationship_integrity ────────────────────────────────────────────────────


async def test_relationship_integrity_pass(validator):
    result = await validator.validate_relationship_integrity(
        expected_relationship_type="DECIDED",
        expected_direction="outgoing",
        relationships=[{"type": "DECIDED", "direction": "->"}],
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_relationship_integrity_wrong_type(validator):
    result = await validator.validate_relationship_integrity(
        expected_relationship_type="DECIDED",
        expected_direction="outgoing",
        relationships=[{"type": "KNOWS", "direction": "outgoing"}],
    )
    assert result.passed is False
    assert result.details["mismatched"][0]["type"] == "KNOWS"
    assert result.error is None


async def test_relationship_integrity_empty_is_quality_failure(validator):
    result = await validator.validate_relationship_integrity(
        expected_relationship_type="DECIDED",
        relationships=[],
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["empty"] is True
    assert result.error is None


async def test_relationship_integrity_requires_query_or_rows(validator, mock_audit):
    result = await validator.validate_relationship_integrity(
        expected_relationship_type="DECIDED",
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── temporal_accuracy ─────────────────────────────────────────────────────────


async def test_temporal_accuracy_pass(validator):
    result = await validator.validate_temporal_accuracy(
        entity_id="claim-1",
        expected_event_sequence=["decision", "outcome"],
        events=[
            {"name": "noise", "timestamp": "2026-01-01T00:00:00Z"},
            {"name": "decision", "timestamp": "2026-01-02T00:00:00Z"},
            {"name": "outcome", "timestamp": "2026-01-03T00:00:00Z"},
        ],
    )
    assert result.passed is True
    assert result.details["order_ok"] is True


async def test_temporal_accuracy_out_of_order(validator):
    result = await validator.validate_temporal_accuracy(
        entity_id="claim-1",
        expected_event_sequence=["decision", "outcome"],
        events=[
            {"name": "outcome", "timestamp": "2026-01-01T00:00:00Z"},
            {"name": "decision", "timestamp": "2026-01-02T00:00:00Z"},
        ],
    )
    assert result.passed is False
    assert result.details["order_ok"] is False
    assert result.error is None


async def test_temporal_accuracy_missing_event(validator):
    result = await validator.validate_temporal_accuracy(
        entity_id="claim-1",
        expected_event_sequence=["decision", "outcome"],
        events=[{"name": "decision", "timestamp": "2026-01-02T00:00:00Z"}],
    )
    assert result.passed is False
    assert result.details["missing"] == ["outcome"]
    assert abs(result.score - 0.5) < 1e-9


async def test_temporal_accuracy_empty_sequence_is_error(validator, mock_audit):
    result = await validator.validate_temporal_accuracy(
        entity_id="claim-1", expected_event_sequence=[]
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── query_correctness ─────────────────────────────────────────────────────────


async def test_query_correctness_pass(validator):
    result = await validator.validate_query_correctness(
        cypher_query="MATCH (n:Claim) RETURN n.id AS id",
        expected_node_ids=["a", "b"],
        rows=[{"id": "a"}, {"id": "b"}],
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_query_correctness_extra_and_missing(validator):
    result = await validator.validate_query_correctness(
        cypher_query="MATCH (n) RETURN n.id AS id",
        expected_node_ids=["a", "b"],
        rows=[{"id": "a"}, {"id": "c"}],
    )
    assert result.passed is False
    assert result.details["missing_ids"] == ["b"]
    assert result.details["extra_ids"] == ["c"]
    assert result.error is None


async def test_query_correctness_empty_expected_is_error(validator, mock_audit):
    result = await validator.validate_query_correctness(
        cypher_query="MATCH (n) RETURN n", expected_node_ids=[]
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── failure_modes ─────────────────────────────────────────────────────────────


async def test_failure_modes_all_surfaced(validator):
    injected = [
        {"kind": "missing", "id": "ghost"},
        {"kind": "duplicate", "id": "dup"},
    ]
    result = await validator.validate_failure_modes(
        injected_anomalies=injected,
        detected=injected,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["silent"] == []


async def test_failure_modes_silent_anomaly(validator):
    result = await validator.validate_failure_modes(
        injected_anomalies=[
            {"kind": "missing", "id": "ghost"},
            {"kind": "duplicate", "id": "dup"},
        ],
        detected=[{"kind": "missing", "id": "ghost"}],
    )
    assert result.passed is False
    assert result.details["silent"] == [["duplicate", "dup"]]
    assert abs(result.score - 0.5) < 1e-9
    assert result.error is None


async def test_failure_modes_empty_injected_is_error(validator, mock_audit):
    result = await validator.validate_failure_modes(injected_anomalies=[])
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── evaluate_all / runner / AuditError ────────────────────────────────────────


async def test_evaluate_dispatches_entity_resolution(validator):
    result = await validator.evaluate(
        metric="entity_resolution",
        entity_sources=[{"entity_key": "x", "email": "a@b.c"}],
        resolved_nodes=[{"entity_key": "x", "id": "n1"}],
    )
    assert result.metric == "entity_resolution"
    assert result.passed is True


async def test_evaluate_all_runs_provided_checks(validator):
    results = await validator.evaluate_all(
        entity_sources=[{"entity_key": "x", "email": "a@b.c"}],
        resolved_nodes=[{"entity_key": "x", "id": "n1"}],
        expected_relationship_type="DECIDED",
        relationships=[{"type": "DECIDED", "direction": "outgoing"}],
        injected_anomalies=[{"kind": "missing", "id": "z"}],
        detected=[{"kind": "missing", "id": "z"}],
    )
    assert [r.metric for r in results] == [
        "entity_resolution",
        "relationship_integrity",
        "failure_modes",
    ]


async def test_audit_error_propagates(validator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError, match="disk full"):
        await validator.validate_entity_resolution(
            entity_sources=[{"entity_key": "x", "email": "a@b.c"}],
            resolved_nodes=[{"entity_key": "x", "id": "n1"}],
        )


async def test_runner_registers_graph(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    integration = MagicMock()
    integration.query = AsyncMock(return_value=[])
    validator = GraphValidator(config, runner.audit_logger, integration=integration)
    runner.register(
        validator,
        metric="entity_resolution",
        entity_sources=[{"entity_key": "x", "email": "a@b.c"}],
        resolved_nodes=[{"entity_key": "x", "id": "n1"}],
    )
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1
