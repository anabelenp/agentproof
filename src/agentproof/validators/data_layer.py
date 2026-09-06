"""DataLayerValidator — PostgreSQL + Redis coordinator.

Dispatches data-layer metrics to `PostgresValidator` and `RedisValidator`,
and scores cache-vs-source consistency across the two stores.

Usage:
    validator = DataLayerValidator(config, audit_logger, postgres=pg, redis=rds)
    result = await validator.validate_cache_source_consistency(
        cache_key="policy:refund",
        lookup_sql="SELECT body FROM policies WHERE id = $1",
        lookup_args=("refund",),
        source_column="body",
    )
"""

from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, DataLayerValidatorError
from agentproof.integrations.postgres import PostgresValidator
from agentproof.integrations.postgres import (
    SUPPORTED_METRICS as POSTGRES_METRICS,
)
from agentproof.integrations.redis import RedisValidator, semantic_equal
from agentproof.integrations.redis import (
    SUPPORTED_METRICS as REDIS_METRICS,
)

SUPPORTED_METRICS = POSTGRES_METRICS + REDIS_METRICS + ("cache_source_consistency",)


class DataLayerValidator(BaseEvaluator):
    """Coordinates PostgreSQL and Redis checks, plus cross-store consistency.

    Child validators write their own audit entries. This class writes an audit
    entry for `cache_source_consistency` and for unknown-metric errors.

    Args:
        config: AgentProofConfig with postgres / redis connection settings.
        audit_logger: AuditLogger that receives every result.
        postgres: Optional PostgresValidator. Created from config if omitted.
        redis: Optional RedisValidator. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        postgres: PostgresValidator | None = None,
        redis: RedisValidator | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._postgres = postgres or PostgresValidator(config, audit_logger)
        self._redis = redis or RedisValidator(config, audit_logger)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "DataLayerValidator"

    @property
    def postgres(self) -> PostgresValidator:
        """The PostgresValidator used for schema, state, and write checks."""
        return self._postgres

    @property
    def redis(self) -> RedisValidator:
        """The RedisValidator used for cache, TTL, and session checks."""
        return self._redis

    async def evaluate(self, **kwargs: Any) -> ValidationResult:
        """Dispatch to PostgreSQL, Redis, or cache-source consistency.

        Args:
            **kwargs: Must include `metric`. Remaining kwargs are forwarded to
                the matching child validator, or to
                `validate_cache_source_consistency`.

        Returns:
            ValidationResult for the requested metric.
        """
        metric = str(kwargs.get("metric") or "schema_integrity").lower().strip()
        if metric in POSTGRES_METRICS:
            return await self._postgres.evaluate(**kwargs)
        if metric in REDIS_METRICS:
            return await self._redis.evaluate(**kwargs)
        if metric == "cache_source_consistency":
            return await self.validate_cache_source_consistency(
                cache_key=str(kwargs.get("cache_key") or kwargs.get("key") or ""),
                lookup_sql=str(kwargs.get("lookup_sql") or ""),
                lookup_args=tuple(kwargs.get("lookup_args") or ()),
                source_column=kwargs.get("source_column"),
                source_fields=kwargs.get("source_fields"),
            )

        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=str(kwargs.get("metric") or ""),
            threshold=0.0,
            details={"supported_metrics": list(SUPPORTED_METRICS)},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
            error=(
                f"Unknown metric {kwargs.get('metric')!r}. "
                f"Expected one of: {', '.join(SUPPORTED_METRICS)}"
            ),
        )
        await self._write_audit(result)
        return result

    async def validate_cache_source_consistency(
        self,
        *,
        cache_key: str,
        lookup_sql: str,
        lookup_args: tuple[Any, ...] | list[Any] = (),
        source_column: str | None = None,
        source_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a Redis value matches the PostgreSQL source of truth.

        A cache miss against a present source row is a stale-miss failure.
        A cache hit against a missing source row is an orphaned-cache failure.
        Both missing is an error (nothing to compare).

        Args:
            cache_key: Redis key holding the cached value.
            lookup_sql: Parameterized SQL that returns the source row.
            lookup_args: Bind args for `lookup_sql`.
            source_column: If set, compare the cache to this column only.
            source_fields: If set, build a dict of these columns and compare
                it to a cached object.

        Returns:
            ValidationResult. score is 1.0 on semantic match, else 0.0.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not cache_key or not cache_key.strip():
                raise DataLayerValidatorError(
                    "cache_key must be a non-empty string",
                    evaluator_name=self.name,
                    metric="cache_source_consistency",
                )
            if not lookup_sql or not lookup_sql.strip():
                raise DataLayerValidatorError(
                    "lookup_sql must be a non-empty string",
                    evaluator_name=self.name,
                    metric="cache_source_consistency",
                )
            cached = await self._redis.integration.get(cache_key)
            row = await self._postgres.integration.fetchrow(
                lookup_sql, *tuple(lookup_args)
            )
            if cached is None and not row:
                raise DataLayerValidatorError(
                    "cache_source_consistency requires a cache value or source row",
                    evaluator_name=self.name,
                    metric="cache_source_consistency",
                )

            source_value: Any
            if row is None:
                source_value = None
            elif source_column:
                source_value = row.get(source_column)
            elif source_fields:
                source_value = {field: row.get(field) for field in source_fields}
            else:
                source_value = row

            cache_miss = cached is None
            source_miss = row is None
            matched = (not cache_miss) and (not source_miss) and semantic_equal(
                cached, source_value
            )
            result = ValidationResult(
                passed=matched,
                score=1.0 if matched else 0.0,
                evaluator_name=self.name,
                metric="cache_source_consistency",
                threshold=1.0,
                details={
                    "cache_key": cache_key,
                    "cache_miss": cache_miss,
                    "source_miss": source_miss,
                    "orphaned_cache": (not cache_miss) and source_miss,
                    "stale_or_mismatch": (not cache_miss)
                    and (not source_miss)
                    and not matched,
                    "matched": matched,
                    "source_column": source_column,
                    "source_fields": list(source_fields or []),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "cache_source_consistency", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(self, **kwargs: Any) -> list[ValidationResult]:
        """Run child evaluate_all plus cache_source_consistency when requested.

        Args:
            **kwargs: Forwarded to PostgresValidator.evaluate_all and
                RedisValidator.evaluate_all. If `cache_key` and `lookup_sql`
                are present, also run cache_source_consistency.

        Returns:
            Combined ValidationResult list. Individual failures do not abort.
        """
        results: list[ValidationResult] = []
        results.extend(await self._postgres.evaluate_all(**kwargs))
        results.extend(await self._redis.evaluate_all(**kwargs))
        if kwargs.get("cache_key") and kwargs.get("lookup_sql"):
            results.append(
                await self.validate_cache_source_consistency(
                    cache_key=str(kwargs["cache_key"]),
                    lookup_sql=str(kwargs["lookup_sql"]),
                    lookup_args=tuple(kwargs.get("lookup_args") or ()),
                    source_column=kwargs.get("source_column"),
                    source_fields=kwargs.get("source_fields"),
                )
            )
        return results
