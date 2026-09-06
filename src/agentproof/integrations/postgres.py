"""PostgreSQL integration and schema / state / transaction validator.

`PostgresIntegration` isolates asyncpg. `PostgresValidator` scores schema
integrity, agent-state persistence, transaction atomicity, silent-write
detection, and write latency against configured SLAs.

Usage:
    pg = PostgresIntegration(config)
    tables = await pg.list_tables()

    validator = PostgresValidator(config, audit_logger, integration=pg)
    result = await validator.validate_schema_integrity(
        expected_tables=["agent_state"],
        expected_columns={"agent_state": ["agent_run_id", "memory"]},
    )
"""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import asyncpg

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import (
    AuditError,
    DataLayerValidatorError,
    IntegrationError,
    RetryExhaustedError,
)
from agentproof.core.retry import RetryConfig, retry_async

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PG_TYPE_ALIASES = {
    "varchar": "character varying",
    "character varying": "character varying",
    "char": "character",
    "character": "character",
    "int": "integer",
    "int4": "integer",
    "integer": "integer",
    "int2": "smallint",
    "smallint": "smallint",
    "int8": "bigint",
    "bigint": "bigint",
    "bool": "boolean",
    "boolean": "boolean",
    "timestamptz": "timestamp with time zone",
    "timestamp with time zone": "timestamp with time zone",
    "timestamp": "timestamp without time zone",
    "timestamp without time zone": "timestamp without time zone",
    "float8": "double precision",
    "double precision": "double precision",
    "float4": "real",
    "real": "real",
    "serial": "integer",
    "bigserial": "bigint",
    "json": "json",
    "jsonb": "jsonb",
    "text": "text",
    "uuid": "uuid",
    "numeric": "numeric",
    "decimal": "numeric",
}

SUPPORTED_METRICS = (
    "schema_integrity",
    "agent_state_persistence",
    "transaction_integrity",
    "write_latency",
    "silent_writes",
)

_TABLES_SQL = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = $1 AND table_type = 'BASE TABLE'
ORDER BY table_name
"""

_COLUMNS_SQL = """
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = $1 AND table_name = $2
ORDER BY ordinal_position
"""

_CONSTRAINTS_SQL = """
SELECT
    tc.constraint_name,
    tc.constraint_type,
    COALESCE(
        array_agg(kcu.column_name ORDER BY kcu.ordinal_position)
            FILTER (WHERE kcu.column_name IS NOT NULL),
        ARRAY[]::text[]
    ) AS columns
FROM information_schema.table_constraints AS tc
LEFT JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
    AND tc.table_name = kcu.table_name
WHERE tc.table_schema = $1 AND tc.table_name = $2
GROUP BY tc.constraint_name, tc.constraint_type
"""


class _InjectedTransactionFailure(Exception):
    """Sentinel that aborts a test transaction so asyncpg rolls it back."""


@dataclass
class ColumnDef:
    """A PostgreSQL column definition from information_schema.

    Args:
        name: Column name.
        data_type: Canonical information_schema data_type.
        is_nullable: True when the column accepts NULL.
    """

    name: str
    data_type: str
    is_nullable: bool = True


@dataclass
class ConstraintDef:
    """A table constraint from information_schema.

    Args:
        name: Constraint name.
        type: Constraint type (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK).
        column_names: Columns covered by the constraint, if known.
    """

    name: str
    type: str
    column_names: list[str] = field(default_factory=list)


@dataclass
class SqlStep:
    """One statement in an atomic multi-step agent write.

    Args:
        sql: Parameterized SQL using asyncpg `$1` placeholders.
        args: Bind parameters, in placeholder order.
    """

    sql: str
    args: tuple[Any, ...] = field(default_factory=tuple)


def quote_ident(name: str) -> str:
    """Validate and double-quote a SQL identifier.

    Args:
        name: Unquoted identifier (table, column, or schema).

    Returns:
        Double-quoted identifier safe to interpolate.

    Raises:
        IntegrationError: If `name` is not a simple identifier.
    """
    if not isinstance(name, str) or not _IDENT.match(name):
        raise IntegrationError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


def normalize_pg_type(data_type: str) -> str:
    """Map common PostgreSQL type aliases to information_schema names.

    Args:
        data_type: Caller or catalog type string.

    Returns:
        Lowercased canonical type name.
    """
    key = data_type.strip().lower()
    return _PG_TYPE_ALIASES.get(key, key)


def record_to_dict(row: Any) -> dict[str, Any]:
    """Convert an asyncpg Record, mapping, or None into a plain dict.

    Args:
        row: Query result row, or None.

    Returns:
        Column-name to value mapping. Empty dict when `row` is None.
    """
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        return {str(key): row[key] for key in row.keys()}
    return dict(row)


def coerce_sql_step(value: SqlStep | dict[str, Any]) -> SqlStep:
    """Build a SqlStep from a dataclass or dict.

    Args:
        value: SqlStep or mapping with `sql` / `args`.

    Returns:
        SqlStep instance.
    """
    if isinstance(value, SqlStep):
        return value
    args = value.get("args") or ()
    if not isinstance(args, tuple):
        args = tuple(args)
    return SqlStep(sql=str(value.get("sql") or ""), args=args)


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _retryable_postgres() -> tuple[type[Exception], ...]:
    """Exceptions that indicate a transient PostgreSQL connectivity failure."""
    retryable: list[type[Exception]] = [
        ConnectionError,
        TimeoutError,
        OSError,
        asyncpg.PostgresConnectionError,
        asyncpg.CannotConnectNowError,
    ]
    too_many = getattr(asyncpg, "TooManyConnectionsError", None)
    if isinstance(too_many, type):
        retryable.append(too_many)
    return tuple(retryable)


class PostgresIntegration:
    """Thin async wrapper around an asyncpg connection pool.

    The pool is created lazily on first query so unit tests can inject a mock
    pool and never open a network connection.

    Args:
        config: AgentProofConfig. `postgres_url` and `postgres_pool_size`
            select the server.
        pool: Optional pre-built asyncpg pool (or test double).
    """

    def __init__(
        self,
        config: AgentProofConfig,
        pool: Any | None = None,
    ) -> None:
        self.config = config
        self._pool = pool
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=_retryable_postgres(),
        )

    async def _ensure_pool(self) -> Any:
        """Return the pool, creating it from `postgres_url` if needed.

        Returns:
            asyncpg pool (or injected test double).

        Raises:
            IntegrationError: Pool creation failed after retries.
        """
        if self._pool is not None:
            return self._pool

        async def _call() -> Any:
            return await asyncpg.create_pool(
                dsn=self.config.postgres_url,
                min_size=1,
                max_size=self.config.postgres_pool_size,
            )

        try:
            self._pool = await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"PostgreSQL pool create failed after {self._retry.max_attempts} "
                f"attempts: {exc.last_exception}"
            ) from exc
        except Exception as exc:
            raise IntegrationError(f"PostgreSQL pool create failed: {exc}") from exc
        return self._pool

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Run a query and return rows as dicts.

        Args:
            sql: Parameterized SQL.
            *args: Bind parameters.

        Returns:
            List of column-name mappings.

        Raises:
            IntegrationError: Empty SQL, or the call failed.
        """
        self._require_sql(sql)

        async def _call() -> list[dict[str, Any]]:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *args)
            return [record_to_dict(row) for row in rows or []]

        return await self._run("fetch", _call)

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """Run a query and return the first row, or None.

        Args:
            sql: Parameterized SQL.
            *args: Bind parameters.

        Returns:
            Column-name mapping, or None when no row matched.

        Raises:
            IntegrationError: Empty SQL, or the call failed.
        """
        self._require_sql(sql)

        async def _call() -> dict[str, Any] | None:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(sql, *args)
            if row is None:
                return None
            return record_to_dict(row)

        return await self._run("fetchrow", _call)

    async def execute(self, sql: str, *args: Any) -> str:
        """Run a statement and return the asyncpg status string.

        Args:
            sql: Parameterized SQL.
            *args: Bind parameters.

        Returns:
            Status such as `"INSERT 0 1"`.

        Raises:
            IntegrationError: Empty SQL, or the call failed.
        """
        self._require_sql(sql)

        async def _call() -> str:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                status = await conn.execute(sql, *args)
            return str(status)

        return await self._run("execute", _call)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Acquire a connection and yield it inside a transaction.

        The transaction commits on clean exit and rolls back on exception.

        Yields:
            An asyncpg connection (or test double).

        Raises:
            IntegrationError: Acquire failed.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        except _InjectedTransactionFailure:
            raise
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"PostgreSQL transaction failed: {exc}") from exc

    async def list_tables(self, schema: str = "public") -> list[str]:
        """List base tables in a schema.

        Args:
            schema: PostgreSQL schema name (default `public`).

        Returns:
            Table names, ordered.

        Raises:
            IntegrationError: Invalid schema name, or the call failed.
        """
        quote_ident(schema)
        rows = await self.fetch(_TABLES_SQL, schema)
        return [str(row.get("table_name") or "") for row in rows if row.get("table_name")]

    async def list_columns(self, table: str, schema: str = "public") -> list[ColumnDef]:
        """List columns for a table.

        Args:
            table: Table name.
            schema: PostgreSQL schema name (default `public`).

        Returns:
            ColumnDef list in ordinal order.

        Raises:
            IntegrationError: Invalid identifiers, or the call failed.
        """
        quote_ident(schema)
        quote_ident(table)
        rows = await self.fetch(_COLUMNS_SQL, schema, table)
        return [
            ColumnDef(
                name=str(row.get("column_name") or ""),
                data_type=normalize_pg_type(str(row.get("data_type") or "")),
                is_nullable=str(row.get("is_nullable") or "YES").upper() != "NO",
            )
            for row in rows
            if row.get("column_name")
        ]

    async def list_constraints(
        self, table: str, schema: str = "public"
    ) -> list[ConstraintDef]:
        """List constraints for a table.

        Args:
            table: Table name.
            schema: PostgreSQL schema name (default `public`).

        Returns:
            ConstraintDef list.

        Raises:
            IntegrationError: Invalid identifiers, or the call failed.
        """
        quote_ident(schema)
        quote_ident(table)
        rows = await self.fetch(_CONSTRAINTS_SQL, schema, table)
        result: list[ConstraintDef] = []
        for row in rows:
            columns = row.get("columns") or []
            if not isinstance(columns, list):
                columns = list(columns)
            result.append(
                ConstraintDef(
                    name=str(row.get("constraint_name") or ""),
                    type=str(row.get("constraint_type") or ""),
                    column_names=[str(col) for col in columns],
                )
            )
        return result

    async def insert(self, table: str, payload: dict[str, Any]) -> str:
        """Insert a row from a column-to-value mapping.

        Args:
            table: Target table.
            payload: Column names to values. Keys must be identifiers.

        Returns:
            asyncpg status string.

        Raises:
            IntegrationError: Empty payload, invalid identifiers, or the call failed.
        """
        sql, args = build_insert(table, payload)
        return await self.execute(sql, *args)

    async def close(self) -> None:
        """Close the underlying pool if it was created."""
        if self._pool is None:
            return
        close = getattr(self._pool, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._pool = None

    async def _run(self, operation: str, call: Any) -> Any:
        """Execute `call` with retry; map failures to IntegrationError.

        Args:
            operation: Short name used in error messages.
            call: Zero-arg async callable.

        Returns:
            The callable's return value.

        Raises:
            IntegrationError: Retries exhausted or a non-retryable error.
        """
        try:
            return await retry_async(self._retry)(call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"PostgreSQL {operation} failed after {self._retry.max_attempts} "
                f"attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"PostgreSQL {operation} failed: {exc}") from exc

    @staticmethod
    def _require_sql(sql: str) -> None:
        """Raise IntegrationError when `sql` is empty or whitespace."""
        if not isinstance(sql, str) or not sql.strip():
            raise IntegrationError("sql must be a non-empty string")


def build_insert(table: str, payload: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a parameterized INSERT from a column mapping.

    Args:
        table: Target table (identifier).
        payload: Column names to values.

    Returns:
        `(sql, args)` suitable for `execute`.

    Raises:
        IntegrationError: Empty payload or invalid identifiers.
    """
    if not payload:
        raise IntegrationError("insert payload must be a non-empty dict")
    columns = list(payload.keys())
    quoted_table = quote_ident(table)
    quoted_cols = ", ".join(quote_ident(col) for col in columns)
    placeholders = ", ".join(f"${index}" for index in range(1, len(columns) + 1))
    sql = f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES ({placeholders})"
    return sql, [payload[col] for col in columns]


def parse_rowcount(status: str) -> int | None:
    """Parse the affected-row count from an asyncpg status string.

    Args:
        status: Status such as `"INSERT 0 1"` or `"UPDATE 3"`.

    Returns:
        Integer row count, or None if the status has no trailing integer.
    """
    if not status:
        return None
    parts = status.strip().split()
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


class PostgresValidator(BaseEvaluator):
    """Scores PostgreSQL schema, state persistence, transactions, and writes.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with postgres URL and write SLA.
        audit_logger: AuditLogger that receives every result.
        integration: Optional PostgresIntegration. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        integration: PostgresIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._integration = integration or PostgresIntegration(config)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "PostgresValidator"

    @property
    def integration(self) -> PostgresIntegration:
        """The PostgresIntegration used for queries and transactions."""
        return self._integration

    async def evaluate(
        self,
        *,
        metric: str = "schema_integrity",
        expected_tables: list[str] | None = None,
        expected_columns: dict[str, Any] | None = None,
        expected_constraints: dict[str, list[str]] | None = None,
        schema: str = "public",
        agent_run_id: str = "",
        expected_state_keys: list[str] | None = None,
        table: str = "agent_state",
        run_id_column: str = "agent_run_id",
        steps: list[SqlStep | dict[str, Any]] | None = None,
        injected_failure_step: int | None = None,
        probe_sql: str = "",
        probe_args: tuple[Any, ...] | list[Any] = (),
        payload: dict[str, Any] | None = None,
        lookup_sql: str = "",
        lookup_args: tuple[Any, ...] | list[Any] = (),
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a PostgreSQL validation method.

        Args:
            metric: One of schema_integrity (default), agent_state_persistence,
                transaction_integrity, write_latency, silent_writes.
            expected_tables: Required tables for schema_integrity.
            expected_columns: Table → column names or name→type map.
            expected_constraints: Table → constraint names or types.
            schema: PostgreSQL schema (default `public`).
            agent_run_id: Run id for agent_state_persistence.
            expected_state_keys: Columns that must be present and non-null.
            table: Target table for state / write checks.
            run_id_column: Column holding `agent_run_id`.
            steps: Atomic SQL steps for transaction_integrity.
            injected_failure_step: Index at which to abort the transaction.
            probe_sql: Query that must be empty after rollback (or non-empty
                after a successful commit).
            probe_args: Bind args for `probe_sql`.
            payload: Row to write for write_latency / silent_writes.
            lookup_sql: Read-back query for silent_writes.
            lookup_args: Bind args for `lookup_sql`.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        if key == "schema_integrity":
            return await self.validate_schema_integrity(
                expected_tables=expected_tables or [],
                expected_columns=expected_columns or {},
                expected_constraints=expected_constraints,
                schema=schema,
            )
        if key == "agent_state_persistence":
            return await self.validate_agent_state_persistence(
                agent_run_id=agent_run_id,
                expected_state_keys=expected_state_keys or [],
                table=table,
                run_id_column=run_id_column,
            )
        if key == "transaction_integrity":
            return await self.validate_transaction_integrity(
                steps=steps or [],
                injected_failure_step=injected_failure_step,
                probe_sql=probe_sql,
                probe_args=tuple(probe_args),
            )
        if key == "write_latency":
            return await self.validate_write_latency(table=table, payload=payload or {})
        if key == "silent_writes":
            return await self.validate_silent_writes(
                table=table,
                payload=payload or {},
                lookup_sql=lookup_sql,
                lookup_args=tuple(lookup_args),
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

    async def validate_schema_integrity(
        self,
        *,
        expected_tables: list[str],
        expected_columns: dict[str, Any] | None = None,
        expected_constraints: dict[str, list[str]] | None = None,
        schema: str = "public",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm tables, columns, and optional constraints match expectations.

        Args:
            expected_tables: Table names that must exist.
            expected_columns: Table → list of column names, or table →
                `{column: type}` map. Types are compared after alias
                normalization (`int` == `integer`).
            expected_constraints: Table → constraint names and/or types
                (`PRIMARY KEY`, `UNIQUE`, …).
            schema: PostgreSQL schema (default `public`).

        Returns:
            ValidationResult. score is passed-checks / total-checks. passed
            only when every expected table, column, and constraint is present.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not expected_tables:
                raise DataLayerValidatorError(
                    "schema_integrity requires non-empty expected_tables",
                    evaluator_name=self.name,
                    metric="schema_integrity",
                )
            actual_tables = set(await self._integration.list_tables(schema))
            missing_tables = [name for name in expected_tables if name not in actual_tables]
            extra_tables = sorted(actual_tables - set(expected_tables))

            missing_columns: dict[str, list[str]] = {}
            type_mismatches: list[dict[str, str]] = []
            missing_constraints: dict[str, list[str]] = {}

            column_checks = 0
            column_hits = 0
            constraint_checks = 0
            constraint_hits = 0
            type_expected = 0

            for table, spec in (expected_columns or {}).items():
                names, types = _split_column_spec(spec)
                column_checks += len(names)
                type_expected += len(types)
                actual_cols = {
                    col.name: col for col in await self._integration.list_columns(table, schema)
                }
                absent = [name for name in names if name not in actual_cols]
                if absent:
                    missing_columns[table] = absent
                column_hits += len(names) - len(absent)
                for name, expected_type in types.items():
                    actual = actual_cols.get(name)
                    if actual is None:
                        continue
                    if normalize_pg_type(expected_type) != actual.data_type:
                        type_mismatches.append(
                            {
                                "table": table,
                                "column": name,
                                "expected": normalize_pg_type(expected_type),
                                "actual": actual.data_type,
                            }
                        )

            for table, expected in (expected_constraints or {}).items():
                constraint_checks += len(expected)
                actual = await self._integration.list_constraints(table, schema)
                tokens = {item.name.lower() for item in actual} | {
                    item.type.lower() for item in actual
                }
                absent = [item for item in expected if item.lower() not in tokens]
                if absent:
                    missing_constraints[table] = absent
                constraint_hits += len(expected) - len(absent)

            table_checks = len(expected_tables)
            table_hits = table_checks - len(missing_tables)
            type_hits = type_expected - len(type_mismatches)
            total = table_checks + column_checks + constraint_checks + type_expected
            passed_checks = table_hits + column_hits + constraint_hits + type_hits
            score = _clamp(passed_checks / total) if total else 1.0
            passed = (
                not missing_tables
                and not missing_columns
                and not type_mismatches
                and not missing_constraints
            )
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="schema_integrity",
                threshold=1.0,
                details={
                    "schema": schema,
                    "expected_tables": list(expected_tables),
                    "actual_tables": sorted(actual_tables),
                    "missing_tables": missing_tables,
                    "extra_tables": extra_tables,
                    "missing_columns": missing_columns,
                    "type_mismatches": type_mismatches,
                    "missing_constraints": missing_constraints,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "schema_integrity", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_agent_state_persistence(
        self,
        *,
        agent_run_id: str,
        expected_state_keys: list[str],
        table: str = "agent_state",
        run_id_column: str = "agent_run_id",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm agent memory for `agent_run_id` is retrievable and complete.

        Args:
            agent_run_id: Run whose state row must exist.
            expected_state_keys: Columns that must be present and non-null.
            table: State table (default `agent_state`).
            run_id_column: Column holding the run id.

        Returns:
            ValidationResult. score is found-keys / expected-keys. passed only
            when the row exists and every key is non-null.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(agent_run_id, "agent_run_id", "agent_state_persistence")
            if not expected_state_keys:
                raise DataLayerValidatorError(
                    "agent_state_persistence requires non-empty expected_state_keys",
                    evaluator_name=self.name,
                    metric="agent_state_persistence",
                )
            quoted_table = quote_ident(table)
            quoted_col = quote_ident(run_id_column)
            sql = f"SELECT * FROM {quoted_table} WHERE {quoted_col} = $1"
            row = await self._integration.fetchrow(sql, agent_run_id)
            if row is None:
                result = ValidationResult(
                    passed=False,
                    score=0.0,
                    evaluator_name=self.name,
                    metric="agent_state_persistence",
                    threshold=1.0,
                    details={
                        "agent_run_id": agent_run_id,
                        "table": table,
                        "run_id_column": run_id_column,
                        "row_found": False,
                        "expected_state_keys": list(expected_state_keys),
                        "present_keys": [],
                        "missing_keys": list(expected_state_keys),
                        "null_keys": [],
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
            else:
                missing = [key for key in expected_state_keys if key not in row]
                null_keys = [
                    key
                    for key in expected_state_keys
                    if key in row and row[key] is None
                ]
                present = [
                    key
                    for key in expected_state_keys
                    if key in row and row[key] is not None
                ]
                score = _clamp(len(present) / len(expected_state_keys))
                result = ValidationResult(
                    passed=not missing and not null_keys,
                    score=score,
                    evaluator_name=self.name,
                    metric="agent_state_persistence",
                    threshold=1.0,
                    details={
                        "agent_run_id": agent_run_id,
                        "table": table,
                        "run_id_column": run_id_column,
                        "row_found": True,
                        "expected_state_keys": list(expected_state_keys),
                        "present_keys": present,
                        "missing_keys": missing,
                        "null_keys": null_keys,
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "agent_state_persistence", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_transaction_integrity(
        self,
        *,
        steps: list[SqlStep | dict[str, Any]],
        injected_failure_step: int | None = None,
        probe_sql: str = "",
        probe_args: tuple[Any, ...] | list[Any] = (),
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a multi-step write is atomic under an injected failure.

        When `injected_failure_step` is set, the transaction aborts at that
        index and `probe_sql` must return no rows (full rollback). When it is
        omitted, all steps commit and `probe_sql` must return at least one row.

        Args:
            steps: Ordered SQL statements that must succeed or roll back together.
            injected_failure_step: Zero-based index at which to raise. None
                means commit the full sequence.
            probe_sql: Query used to detect leftover (or committed) rows.
            probe_args: Bind args for `probe_sql`.

        Returns:
            ValidationResult. score is 1.0 on atomic success, else 0.0.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            parsed = [coerce_sql_step(step) for step in steps]
            if not parsed:
                raise DataLayerValidatorError(
                    "transaction_integrity requires a non-empty steps list",
                    evaluator_name=self.name,
                    metric="transaction_integrity",
                )
            if not probe_sql or not probe_sql.strip():
                raise DataLayerValidatorError(
                    "transaction_integrity requires probe_sql",
                    evaluator_name=self.name,
                    metric="transaction_integrity",
                )
            if injected_failure_step is not None and (
                injected_failure_step < 0 or injected_failure_step > len(parsed)
            ):
                raise DataLayerValidatorError(
                    "injected_failure_step must be in [0, len(steps)]",
                    evaluator_name=self.name,
                    metric="transaction_integrity",
                )

            rolled_back = False
            committed = False
            executed = 0
            try:
                async with self._integration.transaction() as conn:
                    for index, step in enumerate(parsed):
                        if (
                            injected_failure_step is not None
                            and index == injected_failure_step
                        ):
                            raise _InjectedTransactionFailure(
                                f"injected failure at step {injected_failure_step}"
                            )
                        await conn.execute(step.sql, *step.args)
                        executed += 1
                    if (
                        injected_failure_step is not None
                        and injected_failure_step == len(parsed)
                    ):
                        raise _InjectedTransactionFailure(
                            f"injected failure at step {injected_failure_step}"
                        )
                    committed = True
            except _InjectedTransactionFailure:
                rolled_back = True
                committed = False

            remaining = await self._integration.fetch(probe_sql, *tuple(probe_args))
            leftover = len(remaining)
            if injected_failure_step is not None:
                passed = rolled_back and leftover == 0
            else:
                passed = committed and leftover > 0

            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="transaction_integrity",
                threshold=1.0,
                details={
                    "step_count": len(parsed),
                    "executed_steps": executed,
                    "injected_failure_step": injected_failure_step,
                    "rolled_back": rolled_back,
                    "committed": committed,
                    "probe_row_count": leftover,
                    "partial_writes": leftover > 0 and rolled_back,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "transaction_integrity", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_write_latency(
        self,
        *,
        table: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> ValidationResult:
        """Measure a single INSERT against `max_db_write_latency_ms`.

        Args:
            table: Target table.
            payload: Column-to-value mapping to insert.

        Returns:
            ValidationResult. passed if write time <= SLA. score is 1.0 when
            under SLA, otherwise sla / elapsed (clamped).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        sla = self.config.max_db_write_latency_ms
        try:
            if not payload:
                raise DataLayerValidatorError(
                    "write_latency requires a non-empty payload",
                    evaluator_name=self.name,
                    metric="write_latency",
                )
            write_start = self._start_timer()
            status = await self._integration.insert(table, payload)
            write_ms = self._elapsed_ms(write_start)
            passed = write_ms <= sla
            score = 1.0 if passed else _clamp(sla / write_ms if write_ms else 0.0)
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="write_latency",
                threshold=0.0,
                details={
                    "table": table,
                    "write_latency_ms": write_ms,
                    "sla_ms": sla,
                    "status": status,
                    "rowcount": parse_rowcount(status),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "write_latency", exc, threshold=0.0
            )

        await self._write_audit(result)
        return result

    async def validate_silent_writes(
        self,
        *,
        table: str,
        payload: dict[str, Any],
        lookup_sql: str = "",
        lookup_args: tuple[Any, ...] | list[Any] = (),
        **kwargs: Any,
    ) -> ValidationResult:
        """Write a row, then read it back. A missing row is a silent failure.

        A raised write error is surfaced (`silent_write_failure` is False) and
        becomes an error result. A successful status with no retrievable row
        is a quality failure with `silent_write_failure` True.

        Args:
            table: Target table.
            payload: Column-to-value mapping to insert.
            lookup_sql: Optional read-back query. Defaults to SELECT by payload.
            lookup_args: Bind args for `lookup_sql`. Defaults to payload values.

        Returns:
            ValidationResult. passed when the write status reports a row and
            the lookup finds it.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not payload:
                raise DataLayerValidatorError(
                    "silent_writes requires a non-empty payload",
                    evaluator_name=self.name,
                    metric="silent_writes",
                )
            status = await self._integration.insert(table, payload)
            rowcount = parse_rowcount(status)
            sql = lookup_sql
            args: tuple[Any, ...] = tuple(lookup_args)
            if not sql:
                sql, values = _lookup_from_payload(table, payload)
                args = tuple(values)
            row = await self._integration.fetchrow(sql, *args)
            found = row is not None
            status_ok = rowcount is None or rowcount > 0
            silent = status_ok and not found
            passed = found and status_ok
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="silent_writes",
                threshold=1.0,
                details={
                    "table": table,
                    "status": status,
                    "rowcount": rowcount,
                    "row_found": found,
                    "silent_write_failure": silent,
                    "surfaced": not silent,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "silent_writes", exc, threshold=1.0
            )
            result.details = {
                "table": table,
                "silent_write_failure": False,
                "surfaced": True,
            }

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        expected_tables: list[str] | None = None,
        expected_columns: dict[str, Any] | None = None,
        expected_constraints: dict[str, list[str]] | None = None,
        agent_run_id: str = "",
        expected_state_keys: list[str] | None = None,
        table: str = "agent_state",
        steps: list[SqlStep | dict[str, Any]] | None = None,
        injected_failure_step: int | None = None,
        probe_sql: str = "",
        probe_args: tuple[Any, ...] | list[Any] = (),
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ValidationResult]:
        """Run every PostgreSQL check that has enough arguments.

        Args:
            expected_tables / expected_columns / expected_constraints:
                If tables are set, run schema_integrity.
            agent_run_id / expected_state_keys: If both set, run persistence.
            table: State / write table.
            steps / probe_sql: If both set, run transaction_integrity.
            payload: If set, run write_latency and silent_writes.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results: list[ValidationResult] = []
        if expected_tables:
            results.append(
                await self.validate_schema_integrity(
                    expected_tables=expected_tables,
                    expected_columns=expected_columns,
                    expected_constraints=expected_constraints,
                )
            )
        if agent_run_id and expected_state_keys:
            results.append(
                await self.validate_agent_state_persistence(
                    agent_run_id=agent_run_id,
                    expected_state_keys=expected_state_keys,
                    table=table,
                )
            )
        if steps and probe_sql:
            results.append(
                await self.validate_transaction_integrity(
                    steps=steps,
                    injected_failure_step=injected_failure_step,
                    probe_sql=probe_sql,
                    probe_args=tuple(probe_args),
                )
            )
        if payload:
            results.append(
                await self.validate_write_latency(table=table, payload=payload)
            )
            results.append(
                await self.validate_silent_writes(table=table, payload=payload)
            )
        return results

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise DataLayerValidatorError if `value` is empty or whitespace."""
        if not isinstance(value, str) or not value.strip():
            raise DataLayerValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _split_column_spec(spec: Any) -> tuple[list[str], dict[str, str]]:
    """Normalize a column spec into names plus optional type map.

    Args:
        spec: List of column names, or `{name: type}` mapping.

    Returns:
        `(names, types)` where `types` may be empty.
    """
    if isinstance(spec, dict):
        names = [str(name) for name in spec.keys()]
        types = {str(name): str(data_type) for name, data_type in spec.items()}
        return names, types
    return [str(name) for name in spec], {}


def _lookup_from_payload(table: str, payload: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a SELECT that matches every column in `payload`.

    Args:
        table: Target table.
        payload: Column-to-value mapping used as the equality filter.

    Returns:
        `(sql, args)` for `fetchrow`.
    """
    columns = list(payload.keys())
    quoted_table = quote_ident(table)
    clauses = [
        f"{quote_ident(col)} = ${index}" for index, col in enumerate(columns, start=1)
    ]
    sql = f"SELECT * FROM {quoted_table} WHERE {' AND '.join(clauses)}"
    return sql, [payload[col] for col in columns]
