"""Unit tests for RedisIntegration and RedisValidator. All Redis calls mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError
from agentproof.core.runner import TestRunner
from agentproof.integrations.redis import (
    RedisIntegration,
    RedisValidator,
    decode_redis_value,
    encode_redis_value,
    semantic_equal,
)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.ttl = AsyncMock(return_value=3600)
    client.exists = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock(return_value=1)
    client.aclose = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def integration(mock_client) -> RedisIntegration:
    return RedisIntegration(AgentProofConfig(), client=mock_client)


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def validator(mock_audit, integration) -> RedisValidator:
    return RedisValidator(AgentProofConfig(), mock_audit, integration=integration)


# ── encode / decode / semantic_equal ──────────────────────────────────────────


def test_encode_dict_is_json():
    encoded = encode_redis_value({"b": 2, "a": 1})
    assert encoded == '{"a": 1, "b": 2}'


def test_encode_string_passthrough():
    assert encode_redis_value("hello") == "hello"


def test_decode_json_object():
    assert decode_redis_value('{"days": 30}') == {"days": 30}


def test_decode_numeric_string():
    assert decode_redis_value("30") == 30
    assert decode_redis_value("1.5") == 1.5


def test_decode_none_and_bytes():
    assert decode_redis_value(None) is None
    assert decode_redis_value(b'{"ok": true}') == {"ok": True}


def test_semantic_equal_dict_order_and_numbers():
    assert semantic_equal({"days": "30"}, {"days": 30})
    assert semantic_equal({"a": 1, "b": 2}, {"b": 2, "a": 1.0})


def test_semantic_equal_whitespace_strings():
    assert semantic_equal("  refund  ", "refund")


def test_semantic_equal_lists():
    assert semantic_equal(["a", "1"], ["a", 1])
    assert not semantic_equal(["a", "b"], ["b", "a"])


def test_semantic_equal_bool_not_int():
    assert not semantic_equal(True, 1)
    assert semantic_equal(True, True)


# ── RedisIntegration ──────────────────────────────────────────────────────────


def test_init_uses_injected_client(mock_client):
    integration = RedisIntegration(AgentProofConfig(), client=mock_client)
    assert integration._client is mock_client


def test_init_calls_from_url():
    with patch("agentproof.integrations.redis.redis_async.from_url") as mocked:
        mocked.return_value = MagicMock()
        RedisIntegration(AgentProofConfig(redis_url="redis://cache:6379"))
    mocked.assert_called_once()
    assert mocked.call_args.args[0] == "redis://cache:6379"
    assert mocked.call_args.kwargs["decode_responses"] is True


async def test_get_decodes_json(integration, mock_client):
    mock_client.get = AsyncMock(return_value='{"days": 30}')
    assert await integration.get("policy:refund") == {"days": 30}


async def test_get_empty_key_raises(integration):
    with pytest.raises(IntegrationError, match="key"):
        await integration.get("  ")


async def test_set_encodes_and_applies_ttl(integration, mock_client):
    await integration.set("policy:refund", {"days": 30}, ttl_seconds=60)
    mock_client.set.assert_awaited_once()
    args, kwargs = mock_client.set.call_args
    assert args[0] == "policy:refund"
    assert "days" in args[1]
    assert kwargs["ex"] == 60


async def test_set_zero_ttl_omits_ex(integration, mock_client):
    await integration.set("k", "v", ttl_seconds=0)
    kwargs = mock_client.set.call_args.kwargs
    assert "ex" not in kwargs


async def test_delete_returns_count(integration, mock_client):
    mock_client.delete = AsyncMock(return_value=1)
    assert await integration.delete("k") == 1


async def test_ttl_and_exists(integration, mock_client):
    mock_client.ttl = AsyncMock(return_value=12)
    mock_client.exists = AsyncMock(return_value=1)
    assert await integration.ttl("k") == 12
    assert await integration.exists("k") is True


async def test_hgetall_decodes_fields(integration, mock_client):
    mock_client.hgetall = AsyncMock(return_value={"status": '"done"', "turn": "3"})
    result = await integration.hgetall("session:abc")
    assert result["status"] == "done"
    assert result["turn"] == 3


async def test_hset_encodes_mapping(integration, mock_client):
    await integration.hset("session:abc", {"status": "done", "n": 1})
    mapping = mock_client.hset.call_args.kwargs["mapping"]
    assert mapping["status"] == "done"
    assert mapping["n"] == "1"


async def test_hset_empty_mapping_raises(integration):
    with pytest.raises(IntegrationError, match="mapping"):
        await integration.hset("session:abc", {})


async def test_get_retries_connection_error(integration, mock_client):
    mock_client.get = AsyncMock(side_effect=[ConnectionError("down"), "ok"])
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        assert await integration.get("k") == "ok"
    assert mock_client.get.await_count == 2


async def test_get_exhausted_retries_become_integration_error(integration, mock_client):
    mock_client.get = AsyncMock(side_effect=ConnectionError("down"))
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.get("k")


async def test_close_prefers_aclose(integration, mock_client):
    await integration.close()
    mock_client.aclose.assert_awaited_once()
    mock_client.close.assert_not_called()


# ── RedisValidator identity / dispatch ────────────────────────────────────────


def test_name(validator):
    assert validator.name == "RedisValidator"


async def test_unknown_metric(validator, mock_audit):
    result = await validator.evaluate(metric="pubsub")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── cache_correctness ─────────────────────────────────────────────────────────


async def test_cache_correctness_pass(validator, mock_client):
    mock_client.get = AsyncMock(return_value='{"days": 30}')
    result = await validator.validate_cache_correctness(
        key="policy:refund",
        expected_value={"days": 30},
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["matched"] is True
    assert result.error is None


async def test_cache_correctness_mismatch(validator, mock_client):
    mock_client.get = AsyncMock(return_value='{"days": 14}')
    result = await validator.validate_cache_correctness(
        key="policy:refund",
        expected_value={"days": 30},
    )
    assert result.passed is False
    assert result.details["matched"] is False
    assert result.error is None


async def test_cache_correctness_missing_key(validator, mock_client):
    mock_client.get = AsyncMock(return_value=None)
    result = await validator.validate_cache_correctness(
        key="policy:refund",
        expected_value={"days": 30},
    )
    assert result.passed is False
    assert result.details["missing"] is True


async def test_cache_correctness_empty_key_is_error(validator, mock_audit):
    result = await validator.validate_cache_correctness(key="", expected_value=1)
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


async def test_evaluate_dispatches_cache_correctness(validator, mock_client):
    mock_client.get = AsyncMock(return_value="hello")
    result = await validator.evaluate(
        metric="cache_correctness", key="k", expected_value="hello"
    )
    assert result.metric == "cache_correctness"
    assert result.passed is True


# ── ttl_behavior — 10 expiry scenarios ────────────────────────────────────────


async def test_ttl_exact_match(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=3600)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is True
    assert result.details["in_range"] is True


async def test_ttl_within_tolerance_low(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=3597)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is True


async def test_ttl_within_tolerance_high(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=3604)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is True


async def test_ttl_outside_tolerance(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=10)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is False
    assert result.details["in_range"] is False
    assert result.error is None


async def test_ttl_key_missing(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=-2)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is False
    assert result.details["missing"] is True


async def test_ttl_no_expiry_when_expected(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=-1)
    result = await validator.validate_ttl_behavior(key="k", expected_ttl_seconds=3600)
    assert result.passed is False
    assert result.details["no_expiry"] is True


async def test_ttl_custom_tolerance(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=50)
    result = await validator.validate_ttl_behavior(
        key="k", expected_ttl_seconds=60, tolerance_seconds=15
    )
    assert result.passed is True


async def test_ttl_confirm_expiry_gone(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=2)
    mock_client.get = AsyncMock(return_value=None)
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    result = await validator.validate_ttl_behavior(
        key="k",
        expected_ttl_seconds=2,
        tolerance_seconds=0,
        confirm_expiry=True,
        sleeper=fake_sleep,
    )
    assert result.passed is True
    assert result.details["expired_after_wait"] is True
    assert slept == [3]


async def test_ttl_confirm_expiry_still_present(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=2)
    mock_client.get = AsyncMock(return_value="stale")

    async def fake_sleep(_seconds):
        return None

    result = await validator.validate_ttl_behavior(
        key="k",
        expected_ttl_seconds=2,
        tolerance_seconds=0,
        confirm_expiry=True,
        sleeper=fake_sleep,
    )
    assert result.passed is False
    assert result.details["expired_after_wait"] is False


async def test_ttl_zero_expected_with_zero_remaining(validator, mock_client):
    mock_client.ttl = AsyncMock(return_value=0)
    result = await validator.validate_ttl_behavior(
        key="k", expected_ttl_seconds=0, tolerance_seconds=0
    )
    assert result.passed is True


# ── cache_invalidation ────────────────────────────────────────────────────────


async def test_cache_invalidation_deletes_key(validator, mock_client):
    mock_client.get = AsyncMock(side_effect=['{"days": 30}', None])

    async def update():
        return None

    result = await validator.validate_cache_invalidation(
        key="policy:refund",
        upstream_update=update,
        stale_value={"days": 30},
    )
    assert result.passed is True
    assert result.details["invalidated"] is True
    assert result.details["post_update_missing"] is True


async def test_cache_invalidation_replaced_value(validator, mock_client):
    mock_client.get = AsyncMock(side_effect=['{"days": 30}', '{"days": 14}'])
    result = await validator.validate_cache_invalidation(
        key="policy:refund",
        upstream_update=lambda: None,
        stale_value={"days": 30},
    )
    assert result.passed is True
    assert result.details["post_update_missing"] is False


async def test_cache_invalidation_stale_remains(validator, mock_client):
    mock_client.get = AsyncMock(return_value='{"days": 30}')
    result = await validator.validate_cache_invalidation(
        key="policy:refund",
        upstream_update=lambda: None,
    )
    assert result.passed is False
    assert result.details["invalidated"] is False
    assert result.error is None


async def test_cache_invalidation_missing_before_is_error(validator, mock_client, mock_audit):
    mock_client.get = AsyncMock(return_value=None)
    result = await validator.validate_cache_invalidation(
        key="policy:refund",
        upstream_update=lambda: None,
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


async def test_cache_invalidation_requires_update(validator, mock_audit):
    result = await validator.validate_cache_invalidation(key="k")
    assert result.error is not None
    assert "upstream_update" in result.error


# ── session_state ─────────────────────────────────────────────────────────────


async def test_session_state_hash_pass(validator, mock_client):
    mock_client.hgetall = AsyncMock(
        return_value={"status": "done", "turn": "3", "agent": "underwriter"}
    )
    result = await validator.validate_session_state(
        session_id="abc",
        expected_state={"status": "done", "turn": 3, "agent": "underwriter"},
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["source"] == "hash"


async def test_session_state_json_fallback(validator, mock_client):
    mock_client.hgetall = AsyncMock(return_value={})
    mock_client.get = AsyncMock(return_value='{"status": "done", "turn": 3}')
    result = await validator.validate_session_state(
        session_id="abc",
        expected_state={"status": "done", "turn": 3},
    )
    assert result.passed is True
    assert result.details["source"] == "json"
    assert result.details["key"] == "session:abc"


async def test_session_state_partial_mismatch(validator, mock_client):
    mock_client.hgetall = AsyncMock(return_value={"status": "done"})
    result = await validator.validate_session_state(
        session_id="abc",
        expected_state={"status": "done", "turn": 3},
    )
    assert result.passed is False
    assert result.details["missing_keys"] == ["turn"]
    assert abs(result.score - 0.5) < 1e-9


async def test_session_state_empty_expected_is_error(validator, mock_audit):
    result = await validator.validate_session_state(session_id="abc", expected_state={})
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── evaluate_all / runner / AuditError ────────────────────────────────────────


async def test_evaluate_all_runs_provided_checks(validator, mock_client):
    mock_client.get = AsyncMock(return_value="v")
    mock_client.ttl = AsyncMock(return_value=3600)
    results = await validator.evaluate_all(
        key="k",
        expected_value="v",
        expected_ttl_seconds=3600,
    )
    assert [r.metric for r in results] == ["cache_correctness", "ttl_behavior"]


async def test_audit_error_propagates(validator, mock_audit, mock_client):
    mock_client.get = AsyncMock(return_value="v")
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError, match="disk full"):
        await validator.validate_cache_correctness(key="k", expected_value="v")


async def test_runner_registers_redis(tmp_path, mock_client):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    validator = RedisValidator(
        config,
        runner.audit_logger,
        integration=RedisIntegration(config, client=mock_client),
    )
    mock_client.get = AsyncMock(return_value="v")
    runner.register(validator, metric="cache_correctness", key="k", expected_value="v")
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1


async def test_get_failure_becomes_error_result(validator, mock_client, mock_audit):
    mock_client.get = AsyncMock(side_effect=RuntimeError("cluster down"))
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await validator.validate_cache_correctness(key="k", expected_value="v")
    assert result.passed is False
    assert result.error is not None
    assert "cluster down" in result.error
    mock_audit.log.assert_awaited_once()
