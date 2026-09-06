"""Redis integration and cache / TTL / session validator.

`RedisIntegration` isolates the redis.asyncio client. `RedisValidator`
scores cache correctness (semantic equality), TTL behavior, invalidation,
and session-state retrieval.

Usage:
    cache = RedisIntegration(config)
    await cache.set("policy:refund", {"days": 30}, ttl_seconds=3600)

    validator = RedisValidator(config, audit_logger, integration=cache)
    result = await validator.validate_cache_correctness(
        key="policy:refund",
        expected_value={"days": 30},
    )
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from redis import asyncio as redis_async
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

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

SUPPORTED_METRICS = (
    "cache_correctness",
    "ttl_behavior",
    "cache_invalidation",
    "session_state",
)

DEFAULT_SESSION_PREFIX = "session:"


def encode_redis_value(value: Any) -> str:
    """Serialize a Python value for Redis storage.

    Dicts and lists become JSON. Strings pass through. Other JSON-serializable
    values are dumped; everything else becomes `str(value)`.

    Args:
        value: Value to store.

    Returns:
        String payload.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def decode_redis_value(raw: Any) -> Any:
    """Best-effort decode of a Redis payload into a Python value.

    JSON objects/arrays are parsed. Numeric strings become int/float.
    Bytes are decoded as UTF-8. None stays None.

    Args:
        raw: Value returned by the Redis client.

    Returns:
        Decoded Python object.
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare values with JSON / numeric / whitespace normalization.

    Dicts compare by key regardless of insertion order. Lists compare in
    order. `"1"` equals `1` and `1` equals `1.0`.

    Args:
        left: First value (possibly a Redis string).
        right: Second value (possibly a Redis string).

    Returns:
        True when the decoded values are semantically the same.
    """
    a = decode_redis_value(left)
    b = decode_redis_value(right)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(semantic_equal(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(semantic_equal(item_a, item_b) for item_a, item_b in zip(a, b, strict=True))
    if isinstance(a, bool) and isinstance(b, bool):
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class RedisIntegration:
    """Thin async wrapper around redis.asyncio.

    Args:
        config: AgentProofConfig. `redis_url` selects the server.
        client: Optional pre-built redis client (or test double).
    """

    def __init__(
        self,
        config: AgentProofConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client or redis_async.from_url(
            config.redis_url,
            decode_responses=True,
        )
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                RedisConnectionError,
                RedisTimeoutError,
                BusyLoadingError,
            ),
        )

    async def get(self, key: str) -> Any:
        """Get and decode a key.

        Args:
            key: Redis key.

        Returns:
            Decoded value, or None if the key is missing.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)

        async def _call() -> Any:
            return decode_redis_value(await self._client.get(key))

        return await self._run("get", _call)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Set a key, JSON-encoding structured values.

        Args:
            key: Redis key.
            value: Value to store.
            ttl_seconds: Expiry in seconds. None uses `config.redis_ttl_seconds`.
                0 stores the key with no expiry.

        Returns:
            True if Redis acknowledged the write.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)
        encoded = encode_redis_value(value)
        if ttl_seconds is None:
            ttl_seconds = self.config.redis_ttl_seconds

        async def _call() -> Any:
            if ttl_seconds and ttl_seconds > 0:
                return await self._client.set(key, encoded, ex=int(ttl_seconds))
            return await self._client.set(key, encoded)

        result = await self._run("set", _call)
        return bool(result)

    async def delete(self, key: str) -> int:
        """Delete a key.

        Args:
            key: Redis key.

        Returns:
            Number of keys removed.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)

        async def _call() -> Any:
            return await self._client.delete(key)

        result = await self._run("delete", _call)
        return int(result or 0)

    async def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds.

        Redis conventions: -2 missing, -1 no expiry, >= 0 remaining seconds.

        Args:
            key: Redis key.

        Returns:
            Integer TTL.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)

        async def _call() -> Any:
            return await self._client.ttl(key)

        result = await self._run("ttl", _call)
        return int(result)

    async def exists(self, key: str) -> bool:
        """Return True if the key exists.

        Args:
            key: Redis key.

        Returns:
            True when the key is present.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)

        async def _call() -> Any:
            return await self._client.exists(key)

        result = await self._run("exists", _call)
        return bool(result)

    async def hgetall(self, key: str) -> dict[str, Any]:
        """Return a hash as a dict with decoded values.

        Args:
            key: Redis hash key.

        Returns:
            Field-to-decoded-value mapping. Empty if the hash is missing.

        Raises:
            IntegrationError: Empty key, or the call failed.
        """
        self._require_key(key)

        async def _call() -> Any:
            return await self._client.hgetall(key)

        raw = await self._run("hgetall", _call) or {}
        return {str(field): decode_redis_value(value) for field, value in dict(raw).items()}

    async def hset(self, key: str, mapping: dict[str, Any]) -> int:
        """Write a hash from a field mapping.

        Args:
            key: Redis hash key.
            mapping: Field-to-value mapping. Values are encoded.

        Returns:
            Number of fields added.

        Raises:
            IntegrationError: Empty key or mapping, or the call failed.
        """
        self._require_key(key)
        if not mapping:
            raise IntegrationError("hset mapping must be a non-empty dict")
        encoded = {str(field): encode_redis_value(value) for field, value in mapping.items()}

        async def _call() -> Any:
            return await self._client.hset(key, mapping=encoded)

        result = await self._run("hset", _call)
        return int(result or 0)

    async def close(self) -> None:
        """Close the underlying Redis client if it exposes close/aclose()."""
        for name in ("aclose", "close"):
            close = getattr(self._client, name, None)
            if close is None:
                continue
            result = close()
            if hasattr(result, "__await__"):
                await result
            return

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
                f"Redis {operation} failed after {self._retry.max_attempts} "
                f"attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Redis {operation} failed: {exc}") from exc

    @staticmethod
    def _require_key(key: str) -> None:
        """Raise IntegrationError when `key` is empty or whitespace."""
        if not isinstance(key, str) or not key.strip():
            raise IntegrationError("key must be a non-empty string")


class RedisValidator(BaseEvaluator):
    """Scores Redis cache correctness, TTL, invalidation, and session state.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with redis URL and TTL defaults.
        audit_logger: AuditLogger that receives every result.
        integration: Optional RedisIntegration. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        integration: RedisIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._integration = integration or RedisIntegration(config)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "RedisValidator"

    @property
    def integration(self) -> RedisIntegration:
        """The RedisIntegration used for get / set / ttl / hash commands."""
        return self._integration

    async def evaluate(
        self,
        *,
        metric: str = "cache_correctness",
        key: str = "",
        expected_value: Any = None,
        expected_ttl_seconds: int | None = None,
        tolerance_seconds: int | None = None,
        confirm_expiry: bool = False,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        stale_value: Any = None,
        upstream_update: Callable[..., Any] | None = None,
        session_id: str = "",
        expected_state: dict[str, Any] | None = None,
        key_prefix: str = DEFAULT_SESSION_PREFIX,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a Redis validation method.

        Args:
            metric: One of cache_correctness (default), ttl_behavior,
                cache_invalidation, session_state.
            key: Redis key for cache / TTL / invalidation checks.
            expected_value: Source-of-truth value for cache_correctness.
            expected_ttl_seconds: Expected remaining TTL.
            tolerance_seconds: Allowed TTL drift. Defaults to config.
            confirm_expiry: If True, wait for TTL then confirm the key is gone.
            sleeper: Awaitable sleep used when confirm_expiry is True.
            stale_value: Value that must be present before invalidation.
            upstream_update: Callable that mutates the source of truth.
            session_id: Session id for session_state.
            expected_state: Expected session hash / JSON object.
            key_prefix: Prefix prepended to session_id.

        Returns:
            ValidationResult for the requested metric.
        """
        key_name = metric.lower().strip()
        if key_name == "cache_correctness":
            return await self.validate_cache_correctness(
                key=key, expected_value=expected_value
            )
        if key_name == "ttl_behavior":
            return await self.validate_ttl_behavior(
                key=key,
                expected_ttl_seconds=expected_ttl_seconds
                if expected_ttl_seconds is not None
                else self.config.redis_ttl_seconds,
                tolerance_seconds=tolerance_seconds,
                confirm_expiry=confirm_expiry,
                sleeper=sleeper,
            )
        if key_name == "cache_invalidation":
            return await self.validate_cache_invalidation(
                key=key,
                upstream_update=upstream_update,
                stale_value=stale_value,
            )
        if key_name == "session_state":
            return await self.validate_session_state(
                session_id=session_id,
                expected_state=expected_state or {},
                key_prefix=key_prefix,
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

    async def validate_cache_correctness(
        self,
        *,
        key: str,
        expected_value: Any,
        **kwargs: Any,
    ) -> ValidationResult:
        """Compare a cached value to a source-of-truth using semantic equality.

        Args:
            key: Redis key.
            expected_value: Value the cache should hold. Compared after JSON
                and numeric normalization, not raw string identity.

        Returns:
            ValidationResult. score is 1.0 on match, else 0.0.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(key, "key", "cache_correctness")
            actual = await self._integration.get(key)
            missing = actual is None
            matched = (not missing) and semantic_equal(actual, expected_value)
            result = ValidationResult(
                passed=matched,
                score=1.0 if matched else 0.0,
                evaluator_name=self.name,
                metric="cache_correctness",
                threshold=1.0,
                details={
                    "key": key,
                    "missing": missing,
                    "matched": matched,
                    "actual": actual,
                    "expected": decode_redis_value(expected_value)
                    if isinstance(expected_value, str | bytes)
                    else expected_value,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "cache_correctness", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_ttl_behavior(
        self,
        *,
        key: str,
        expected_ttl_seconds: int,
        tolerance_seconds: int | None = None,
        confirm_expiry: bool = False,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm remaining TTL is within tolerance; optionally wait for expiry.

        `confirm_expiry` is opt-in so default runs never sleep for a full TTL.
        Pass a mocked `sleeper` in unit tests.

        Args:
            key: Redis key.
            expected_ttl_seconds: Expected remaining TTL.
            tolerance_seconds: Allowed absolute drift. Defaults to
                `config.ttl_tolerance_seconds` (5).
            confirm_expiry: If True, sleep remaining TTL + 1s and confirm gone.
            sleeper: Awaitable sleep (default `asyncio.sleep`).

        Returns:
            ValidationResult. passed when TTL is in range (and the key is gone
            after the wait when confirm_expiry is True).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        tolerance = (
            self.config.ttl_tolerance_seconds
            if tolerance_seconds is None
            else tolerance_seconds
        )
        try:
            self._require_text(key, "key", "ttl_behavior")
            if expected_ttl_seconds < 0:
                raise DataLayerValidatorError(
                    "expected_ttl_seconds must be >= 0",
                    evaluator_name=self.name,
                    metric="ttl_behavior",
                )
            remaining = await self._integration.ttl(key)
            missing = remaining == -2
            persistent = remaining == -1
            in_range = (
                not missing
                and not persistent
                and abs(remaining - expected_ttl_seconds) <= tolerance
            )
            expired_ok: bool | None = None
            if confirm_expiry:
                wait_for = max(remaining, 0) + 1
                await (sleeper or asyncio.sleep)(wait_for)
                expired_ok = await self._integration.get(key) is None

            passed = in_range and (expired_ok is not False)
            if missing or (persistent and expected_ttl_seconds > 0):
                passed = False
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="ttl_behavior",
                threshold=1.0,
                details={
                    "key": key,
                    "expected_ttl_seconds": expected_ttl_seconds,
                    "actual_ttl_seconds": remaining,
                    "tolerance_seconds": tolerance,
                    "missing": missing,
                    "no_expiry": persistent,
                    "in_range": in_range,
                    "confirm_expiry": confirm_expiry,
                    "expired_after_wait": expired_ok,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "ttl_behavior", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_cache_invalidation(
        self,
        *,
        key: str,
        upstream_update: Callable[..., Any] | None = None,
        stale_value: Any = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a cache entry is evicted or replaced after an upstream change.

        Args:
            key: Redis key.
            upstream_update: Sync or async callable that mutates the source.
                Required. After it returns, the cached value must be gone or
                different from the pre-update (stale) value.
            stale_value: If provided, the pre-update cache must match it.

        Returns:
            ValidationResult. score is 1.0 when stale data is gone.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(key, "key", "cache_invalidation")
            if upstream_update is None:
                raise DataLayerValidatorError(
                    "cache_invalidation requires upstream_update",
                    evaluator_name=self.name,
                    metric="cache_invalidation",
                )
            before = await self._integration.get(key)
            if before is None:
                raise DataLayerValidatorError(
                    "cache_invalidation requires the key to be present before update",
                    evaluator_name=self.name,
                    metric="cache_invalidation",
                )
            stale_ok = stale_value is None or semantic_equal(before, stale_value)
            if not stale_ok:
                result = ValidationResult(
                    passed=False,
                    score=0.0,
                    evaluator_name=self.name,
                    metric="cache_invalidation",
                    threshold=1.0,
                    details={
                        "key": key,
                        "pre_update_present": True,
                        "stale_matched": False,
                        "invalidated": False,
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
            else:
                outcome = upstream_update()
                if hasattr(outcome, "__await__"):
                    await outcome
                after = await self._integration.get(key)
                invalidated = after is None or not semantic_equal(after, before)
                result = ValidationResult(
                    passed=invalidated,
                    score=1.0 if invalidated else 0.0,
                    evaluator_name=self.name,
                    metric="cache_invalidation",
                    threshold=1.0,
                    details={
                        "key": key,
                        "pre_update_present": True,
                        "stale_matched": True,
                        "invalidated": invalidated,
                        "post_update_missing": after is None,
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "cache_invalidation", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_session_state(
        self,
        *,
        session_id: str,
        expected_state: dict[str, Any],
        key_prefix: str = DEFAULT_SESSION_PREFIX,
        **kwargs: Any,
    ) -> ValidationResult:
        """Compare agent session state in Redis to an expected mapping.

        Tries `HGETALL` first; if the hash is empty, falls back to a JSON GET
        at `{key_prefix}{session_id}`.

        Args:
            session_id: Session identifier.
            expected_state: Field-to-value mapping that must match.
            key_prefix: Prefix prepended to session_id (default `session:`).

        Returns:
            ValidationResult. score is matching-fields / expected-fields.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(session_id, "session_id", "session_state")
            if not expected_state:
                raise DataLayerValidatorError(
                    "session_state requires a non-empty expected_state",
                    evaluator_name=self.name,
                    metric="session_state",
                )
            key = f"{key_prefix}{session_id}"
            actual = await self._integration.hgetall(key)
            source = "hash"
            if not actual:
                blob = await self._integration.get(key)
                source = "json"
                if isinstance(blob, dict):
                    actual = blob
                elif blob is None:
                    actual = {}
                else:
                    actual = {"value": blob}

            missing = [field for field in expected_state if field not in actual]
            mismatched = [
                field
                for field, value in expected_state.items()
                if field in actual and not semantic_equal(actual[field], value)
            ]
            matched = [
                field
                for field in expected_state
                if field in actual and field not in mismatched
            ]
            score = _clamp(len(matched) / len(expected_state))
            result = ValidationResult(
                passed=not missing and not mismatched,
                score=score,
                evaluator_name=self.name,
                metric="session_state",
                threshold=1.0,
                details={
                    "session_id": session_id,
                    "key": key,
                    "source": source,
                    "expected_keys": list(expected_state.keys()),
                    "present_keys": list(actual.keys()),
                    "missing_keys": missing,
                    "mismatched_keys": mismatched,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "session_state", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        key: str = "",
        expected_value: Any = None,
        expected_ttl_seconds: int | None = None,
        upstream_update: Callable[..., Any] | None = None,
        session_id: str = "",
        expected_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ValidationResult]:
        """Run every Redis check that has enough arguments.

        Args:
            key / expected_value: If both set, run cache_correctness.
            key / expected_ttl_seconds: If key is set, run ttl_behavior.
            key / upstream_update: If both set, run cache_invalidation.
            session_id / expected_state: If both set, run session_state.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results: list[ValidationResult] = []
        if key and expected_value is not None:
            results.append(
                await self.validate_cache_correctness(
                    key=key, expected_value=expected_value
                )
            )
        if key and expected_ttl_seconds is not None:
            results.append(
                await self.validate_ttl_behavior(
                    key=key, expected_ttl_seconds=expected_ttl_seconds
                )
            )
        if key and upstream_update is not None:
            results.append(
                await self.validate_cache_invalidation(
                    key=key, upstream_update=upstream_update
                )
            )
        if session_id and expected_state:
            results.append(
                await self.validate_session_state(
                    session_id=session_id, expected_state=expected_state
                )
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
