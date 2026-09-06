"""Unit tests for PostgresIntegration, PostgresValidator, and DataLayerValidator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, DataLayerValidatorError, IntegrationError
from agentproof.core.runner import TestRunner
from agentproof.integrations.postgres import (
    ColumnDef,
    ConstraintDef,
    PostgresIntegration,
    PostgresValidator,
    SqlStep,
    build_insert,
    normalize_pg_type,
    parse_rowcount,
    quote_ident,
    record_to_dict,
)
from agentproof.integrations.redis import RedisIntegration, RedisValidator
from agentproof.validators.data_layer import DataLayerValidator


def _async_cm(value=None):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _pool_and_conn():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.transaction = MagicMock(return_value=_async_cm(None))
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_async_cm(conn))
    pool.close = AsyncMock()
    return pool, conn


@pytest.fixture
def pool_conn():
    return _pool_and_conn()


@pytest.fixture
def integration(pool_conn) -> PostgresIntegration:
    pool, _ = pool_conn
    return PostgresIntegration(AgentProofConfig(), pool=pool)


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def validator(mock_audit, integration) -> PostgresValidator:
    return PostgresValidator(AgentProofConfig(), mock_audit, integration=integration)


@pytest.fixture
def redis_client():
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.ttl = AsyncMock(return_value=3600)
    client.exists = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock(return_value=1)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def data_layer(mock_audit, integration, redis_client) -> DataLayerValidator:
    pg = PostgresValidator(AgentProofConfig(), mock_audit, integration=integration)
    rds = RedisValidator(
        AgentProofConfig(),
        mock_audit,
        integration=RedisIntegration(AgentProofConfig(), client=redis_client),
    )
    return DataLayerValidator(AgentProofConfig(), mock_audit, postgres=pg, redis=rds)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_quote_ident_wraps_valid_name():
    assert quote_ident("agent_state") == '"agent_state"'


def test_quote_ident_rejects_injection():
    with pytest.raises(IntegrationError, match="invalid SQL identifier"):
        quote_ident("agent_state; DROP TABLE users")


def test_normalize_pg_type_aliases():
    assert normalize_pg_type("int") == "integer"
    assert normalize_pg_type("VARCHAR") == "character varying"
    assert normalize_pg_type("timestamptz") == "timestamp with time zone"


def test_record_to_dict_from_mapping_and_none():
    assert record_to_dict({"id": 1}) == {"id": 1}
    assert record_to_dict(None) == {}


def test_record_to_dict_from_keys_object():
    class Row:
        def keys(self):
            return ["a"]

        def __getitem__(self, key):
            return 7

    assert record_to_dict(Row()) == {"a": 7}


def test_build_insert():
    sql, args = build_insert("agent_state", {"id": "r1", "memory": "x"})
    assert sql == 'INSERT INTO "agent_state" ("id", "memory") VALUES ($1, $2)'
    assert args == ["r1", "x"]


def test_build_insert_empty_raises():
    with pytest.raises(IntegrationError, match="payload"):
        build_insert("agent_state", {})


def test_parse_rowcount():
    assert parse_rowcount("INSERT 0 1") == 1
    assert parse_rowcount("UPDATE 3") == 3
    assert parse_rowcount("") is None


# ── PostgresIntegration ───────────────────────────────────────────────────────


async def test_fetch_returns_dicts(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    rows = await integration.fetch("SELECT 1")
    assert rows == [{"table_name": "agent_state"}]


async def test_fetch_empty_sql_raises(integration):
    with pytest.raises(IntegrationError, match="sql"):
        await integration.fetch("  ")


async def test_fetchrow_none(integration, pool_conn):
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value=None)
    assert await integration.fetchrow("SELECT 1") is None


async def test_execute_returns_status(integration, pool_conn):
    _, conn = pool_conn
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    assert await integration.execute("INSERT INTO t VALUES ($1)", "x") == "INSERT 0 1"


async def test_list_tables(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(
        return_value=[{"table_name": "agent_state"}, {"table_name": "events"}]
    )
    assert await integration.list_tables() == ["agent_state", "events"]


async def test_list_columns(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(
        return_value=[
            {"column_name": "id", "data_type": "uuid", "is_nullable": "NO"},
            {"column_name": "memory", "data_type": "jsonb", "is_nullable": "YES"},
        ]
    )
    cols = await integration.list_columns("agent_state")
    assert cols[0] == ColumnDef(name="id", data_type="uuid", is_nullable=False)
    assert cols[1].data_type == "jsonb"


async def test_list_constraints(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(
        return_value=[
            {
                "constraint_name": "agent_state_pkey",
                "constraint_type": "PRIMARY KEY",
                "columns": ["id"],
            }
        ]
    )
    constraints = await integration.list_constraints("agent_state")
    assert constraints[0] == ConstraintDef(
        name="agent_state_pkey", type="PRIMARY KEY", column_names=["id"]
    )


async def test_insert_builds_sql(integration, pool_conn):
    _, conn = pool_conn
    await integration.insert("agent_state", {"id": "r1"})
    conn.execute.assert_awaited()
    sql = conn.execute.call_args.args[0]
    assert 'INSERT INTO "agent_state"' in sql


async def test_fetch_retries_connection_error(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(side_effect=[ConnectionError("down"), [{"n": 1}]])
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        rows = await integration.fetch("SELECT 1")
    assert rows == [{"n": 1}]
    assert conn.fetch.await_count == 2


async def test_fetch_exhausted_retries_become_integration_error(integration, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(side_effect=ConnectionError("down"))
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.fetch("SELECT 1")


async def test_close_delegates_to_pool(integration, pool_conn):
    pool, _ = pool_conn
    await integration.close()
    pool.close.assert_awaited_once()


async def test_transaction_yields_connection(integration, pool_conn):
    _, conn = pool_conn
    async with integration.transaction() as acquired:
        assert acquired is conn


# ── PostgresValidator identity / dispatch ─────────────────────────────────────


def test_name(validator):
    assert validator.name == "PostgresValidator"


async def test_unknown_metric_is_error_result(validator, mock_audit):
    result = await validator.evaluate(metric="vacuum")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── schema_integrity ──────────────────────────────────────────────────────────


async def test_schema_integrity_pass(validator, pool_conn):
    _, conn = pool_conn

    async def fetch(sql, *args):
        if "information_schema.tables" in sql:
            return [{"table_name": "agent_state"}]
        if "information_schema.columns" in sql:
            return [
                {"column_name": "agent_run_id", "data_type": "uuid", "is_nullable": "NO"},
                {"column_name": "memory", "data_type": "jsonb", "is_nullable": "YES"},
            ]
        if "table_constraints" in sql:
            return [
                {
                    "constraint_name": "agent_state_pkey",
                    "constraint_type": "PRIMARY KEY",
                    "columns": ["agent_run_id"],
                }
            ]
        return []

    conn.fetch = AsyncMock(side_effect=fetch)
    result = await validator.validate_schema_integrity(
        expected_tables=["agent_state"],
        expected_columns={"agent_state": {"agent_run_id": "uuid", "memory": "jsonb"}},
        expected_constraints={"agent_state": ["PRIMARY KEY"]},
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.metric == "schema_integrity"
    assert result.error is None


async def test_schema_integrity_missing_table(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "other"}])
    result = await validator.validate_schema_integrity(expected_tables=["agent_state"])
    assert result.passed is False
    assert result.details["missing_tables"] == ["agent_state"]
    assert result.error is None


async def test_schema_integrity_missing_column(validator, pool_conn):
    _, conn = pool_conn

    async def fetch(sql, *args):
        if "information_schema.tables" in sql:
            return [{"table_name": "agent_state"}]
        if "information_schema.columns" in sql:
            return [
                {"column_name": "agent_run_id", "data_type": "uuid", "is_nullable": "NO"}
            ]
        return []

    conn.fetch = AsyncMock(side_effect=fetch)
    result = await validator.validate_schema_integrity(
        expected_tables=["agent_state"],
        expected_columns={"agent_state": ["agent_run_id", "memory"]},
    )
    assert result.passed is False
    assert result.details["missing_columns"]["agent_state"] == ["memory"]


async def test_schema_integrity_type_mismatch(validator, pool_conn):
    _, conn = pool_conn

    async def fetch(sql, *args):
        if "information_schema.tables" in sql:
            return [{"table_name": "agent_state"}]
        if "information_schema.columns" in sql:
            return [
                {"column_name": "memory", "data_type": "text", "is_nullable": "YES"}
            ]
        return []

    conn.fetch = AsyncMock(side_effect=fetch)
    result = await validator.validate_schema_integrity(
        expected_tables=["agent_state"],
        expected_columns={"agent_state": {"memory": "jsonb"}},
    )
    assert result.passed is False
    assert result.details["type_mismatches"][0]["expected"] == "jsonb"
    assert result.details["type_mismatches"][0]["actual"] == "text"


async def test_schema_integrity_empty_tables_is_error(validator, mock_audit):
    result = await validator.validate_schema_integrity(expected_tables=[])
    assert result.error is not None
    assert "expected_tables" in result.error
    mock_audit.log.assert_awaited_once()


async def test_evaluate_dispatches_schema_integrity(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    result = await validator.evaluate(
        metric="schema_integrity", expected_tables=["agent_state"]
    )
    assert result.metric == "schema_integrity"
    assert result.passed is True


# ── agent_state_persistence ───────────────────────────────────────────────────


async def test_state_persistence_pass(validator, pool_conn):
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(
        return_value={"agent_run_id": "run-1", "memory": {"k": 1}, "status": "done"}
    )
    result = await validator.validate_agent_state_persistence(
        agent_run_id="run-1",
        expected_state_keys=["memory", "status"],
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["row_found"] is True


async def test_state_persistence_missing_row(validator, pool_conn):
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value=None)
    result = await validator.validate_agent_state_persistence(
        agent_run_id="run-1",
        expected_state_keys=["memory"],
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["row_found"] is False
    assert result.error is None


async def test_state_persistence_null_key(validator, pool_conn):
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value={"memory": None, "status": "done"})
    result = await validator.validate_agent_state_persistence(
        agent_run_id="run-1",
        expected_state_keys=["memory", "status"],
    )
    assert result.passed is False
    assert result.details["null_keys"] == ["memory"]
    assert abs(result.score - 0.5) < 1e-9


async def test_state_persistence_empty_run_id_is_error(validator, mock_audit):
    result = await validator.validate_agent_state_persistence(
        agent_run_id="  ",
        expected_state_keys=["memory"],
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── transaction_integrity ─────────────────────────────────────────────────────


async def test_transaction_rollback_on_injected_failure(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[])
    result = await validator.validate_transaction_integrity(
        steps=[
            SqlStep("INSERT INTO agent_state (id) VALUES ($1)", ("r1",)),
            SqlStep("INSERT INTO events (id) VALUES ($1)", ("r1",)),
        ],
        injected_failure_step=1,
        probe_sql="SELECT * FROM agent_state WHERE id = $1",
        probe_args=("r1",),
    )
    assert result.passed is True
    assert result.details["rolled_back"] is True
    assert result.details["partial_writes"] is False
    assert result.details["executed_steps"] == 1


async def test_transaction_detects_partial_writes(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"id": "r1"}])
    result = await validator.validate_transaction_integrity(
        steps=[
            SqlStep("INSERT INTO agent_state (id) VALUES ($1)", ("r1",)),
            SqlStep("INSERT INTO events (id) VALUES ($1)", ("r1",)),
        ],
        injected_failure_step=1,
        probe_sql="SELECT * FROM agent_state",
    )
    assert result.passed is False
    assert result.details["partial_writes"] is True
    assert result.error is None


async def test_transaction_commit_path(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"id": "r1"}])
    result = await validator.validate_transaction_integrity(
        steps=[{"sql": "INSERT INTO agent_state (id) VALUES ($1)", "args": ["r1"]}],
        probe_sql="SELECT * FROM agent_state WHERE id = $1",
        probe_args=("r1",),
    )
    assert result.passed is True
    assert result.details["committed"] is True


async def test_transaction_missing_steps_is_error(validator, mock_audit):
    result = await validator.validate_transaction_integrity(steps=[], probe_sql="SELECT 1")
    assert result.error is not None
    assert "steps" in result.error
    mock_audit.log.assert_awaited_once()


# ── write_latency ─────────────────────────────────────────────────────────────


async def test_write_latency_under_sla(validator):
    result = await validator.validate_write_latency(
        table="agent_state", payload={"id": "r1"}
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["sla_ms"] == 200.0


async def test_write_latency_over_sla(validator):
    validator._elapsed_ms = lambda start: 400.0  # type: ignore[method-assign]
    result = await validator.validate_write_latency(
        table="agent_state", payload={"id": "r1"}
    )
    assert result.passed is False
    assert abs(result.score - 200.0 / 400.0) < 1e-9


async def test_write_latency_empty_payload_is_error(validator, mock_audit):
    result = await validator.validate_write_latency(table="agent_state", payload={})
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── silent_writes ─────────────────────────────────────────────────────────────


async def test_silent_writes_round_trip(validator, pool_conn):
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value={"id": "r1"})
    result = await validator.validate_silent_writes(
        table="agent_state", payload={"id": "r1"}
    )
    assert result.passed is True
    assert result.details["silent_write_failure"] is False
    assert result.details["row_found"] is True


async def test_silent_writes_detects_dropped_row(validator, pool_conn):
    _, conn = pool_conn
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchrow = AsyncMock(return_value=None)
    result = await validator.validate_silent_writes(
        table="agent_state", payload={"id": "r1"}
    )
    assert result.passed is False
    assert result.details["silent_write_failure"] is True
    assert result.error is None


async def test_silent_writes_surfaces_execute_error(validator, pool_conn, mock_audit):
    _, conn = pool_conn
    conn.execute = AsyncMock(side_effect=RuntimeError("permission denied"))
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await validator.validate_silent_writes(
            table="agent_state", payload={"id": "r1"}
        )
    assert result.passed is False
    assert result.error is not None
    assert result.details["silent_write_failure"] is False
    assert result.details["surfaced"] is True
    mock_audit.log.assert_awaited_once()


# ── evaluate_all / runner / AuditError ────────────────────────────────────────


async def test_evaluate_all_runs_provided_checks(validator, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    conn.fetchrow = AsyncMock(return_value={"memory": "x", "status": "ok"})
    results = await validator.evaluate_all(
        expected_tables=["agent_state"],
        agent_run_id="run-1",
        expected_state_keys=["memory", "status"],
    )
    assert [r.metric for r in results] == [
        "schema_integrity",
        "agent_state_persistence",
    ]


async def test_audit_error_propagates(validator, mock_audit, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError, match="disk full"):
        await validator.validate_schema_integrity(expected_tables=["agent_state"])


async def test_runner_registers_postgres(tmp_path, pool_conn):
    pool, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    validator = PostgresValidator(
        config, runner.audit_logger, integration=PostgresIntegration(config, pool=pool)
    )
    runner.register(
        validator,
        metric="schema_integrity",
        expected_tables=["agent_state"],
    )
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1


def test_data_layer_validator_error_is_evaluator_error():
    err = DataLayerValidatorError("x", evaluator_name="PostgresValidator", metric="schema_integrity")
    assert err.metric == "schema_integrity"


# ── DataLayerValidator ────────────────────────────────────────────────────────


def test_data_layer_name(data_layer):
    assert data_layer.name == "DataLayerValidator"


async def test_data_layer_dispatches_postgres(data_layer, pool_conn):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    result = await data_layer.evaluate(
        metric="schema_integrity", expected_tables=["agent_state"]
    )
    assert result.evaluator_name == "PostgresValidator"
    assert result.passed is True


async def test_data_layer_dispatches_redis(data_layer, redis_client):
    redis_client.get = AsyncMock(return_value='{"days": 30}')
    result = await data_layer.evaluate(
        metric="cache_correctness",
        key="policy:refund",
        expected_value={"days": 30},
    )
    assert result.evaluator_name == "RedisValidator"
    assert result.passed is True


async def test_data_layer_unknown_metric(data_layer, mock_audit):
    result = await data_layer.evaluate(metric="firestore")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited()


async def test_cache_source_consistency_pass(data_layer, pool_conn, redis_client):
    redis_client.get = AsyncMock(return_value='{"days": 30}')
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value={"body": {"days": 30}})
    result = await data_layer.validate_cache_source_consistency(
        cache_key="policy:refund",
        lookup_sql="SELECT body FROM policies WHERE id = $1",
        lookup_args=("refund",),
        source_column="body",
    )
    assert result.passed is True
    assert result.metric == "cache_source_consistency"
    assert result.details["matched"] is True


async def test_cache_source_consistency_mismatch(data_layer, pool_conn, redis_client):
    redis_client.get = AsyncMock(return_value='{"days": 14}')
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value={"body": {"days": 30}})
    result = await data_layer.validate_cache_source_consistency(
        cache_key="policy:refund",
        lookup_sql="SELECT body FROM policies WHERE id = $1",
        lookup_args=("refund",),
        source_column="body",
    )
    assert result.passed is False
    assert result.details["stale_or_mismatch"] is True
    assert result.error is None


async def test_cache_source_consistency_orphaned_cache(data_layer, pool_conn, redis_client):
    redis_client.get = AsyncMock(return_value='{"days": 30}')
    _, conn = pool_conn
    conn.fetchrow = AsyncMock(return_value=None)
    result = await data_layer.validate_cache_source_consistency(
        cache_key="policy:refund",
        lookup_sql="SELECT body FROM policies WHERE id = $1",
        lookup_args=("refund",),
    )
    assert result.passed is False
    assert result.details["orphaned_cache"] is True


async def test_cache_source_consistency_both_missing_is_error(data_layer, mock_audit):
    result = await data_layer.validate_cache_source_consistency(
        cache_key="policy:refund",
        lookup_sql="SELECT 1",
    )
    assert result.error is not None
    mock_audit.log.assert_awaited()


async def test_data_layer_evaluate_all(data_layer, pool_conn, redis_client):
    _, conn = pool_conn
    conn.fetch = AsyncMock(return_value=[{"table_name": "agent_state"}])
    conn.fetchrow = AsyncMock(return_value={"body": {"days": 30}})
    redis_client.get = AsyncMock(return_value='{"days": 30}')
    results = await data_layer.evaluate_all(
        expected_tables=["agent_state"],
        key="policy:refund",
        expected_value={"days": 30},
        cache_key="policy:refund",
        lookup_sql="SELECT body FROM policies WHERE id = $1",
        lookup_args=("refund",),
        source_column="body",
    )
    metrics = [r.metric for r in results]
    assert "schema_integrity" in metrics
    assert "cache_correctness" in metrics
    assert "cache_source_consistency" in metrics
