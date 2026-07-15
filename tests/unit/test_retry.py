"""Unit tests for RetryConfig and retry_async decorator."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agentproof.core.errors import RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async


# ── Happy paths ───────────────────────────────────────────────────────────────


async def test_success_on_first_attempt():
    call_count = 0

    @retry_async()
    async def always_works():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await always_works()
    assert result == "ok"
    assert call_count == 1


async def test_retry_succeeds_on_second_attempt():
    call_count = 0
    config = RetryConfig(max_attempts=3, base_delay=0.001)

    @retry_async(config)
    async def fails_once():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("first try")
        return "recovered"

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await fails_once()

    assert result == "recovered"
    assert call_count == 2


async def test_preserves_complex_return_value():
    @retry_async()
    async def returns_dict():
        return {"score": 0.9, "tags": ["a", "b"], "nested": {"ok": True}}

    result = await returns_dict()
    assert result == {"score": 0.9, "tags": ["a", "b"], "nested": {"ok": True}}


# ── Exhaustion ────────────────────────────────────────────────────────────────


async def test_raises_retry_exhausted_after_max_attempts():
    config = RetryConfig(max_attempts=3, base_delay=0.001)

    @retry_async(config)
    async def always_fails():
        raise RuntimeError("permanent failure")

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fails()

    assert exc_info.value.last_exception is not None
    assert isinstance(exc_info.value.last_exception, RuntimeError)


async def test_attempt_count_equals_max_attempts():
    call_count = 0
    config = RetryConfig(max_attempts=4, base_delay=0.001)

    @retry_async(config)
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RetryExhaustedError):
            await always_fails()

    assert call_count == 4


async def test_sleep_called_n_minus_one_times():
    config = RetryConfig(max_attempts=3, base_delay=1.0, jitter=False)
    sleep_calls: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    @retry_async(config)
    async def always_fails():
        raise ValueError("fail")

    with patch("agentproof.core.retry.asyncio.sleep", side_effect=capture_sleep):
        with pytest.raises(RetryExhaustedError):
            await always_fails()

    assert len(sleep_calls) == 2  # 3 attempts → 2 sleeps between them


# ── Exception filtering ───────────────────────────────────────────────────────


async def test_does_not_retry_non_configured_exception():
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.001,
        retryable_exceptions=(ValueError,),
    )
    call_count = 0

    @retry_async(config)
    async def raises_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("not retryable")

    with pytest.raises(TypeError):
        await raises_type_error()

    assert call_count == 1  # no retry attempted


async def test_retries_only_matching_exception():
    call_count = 0
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.001,
        retryable_exceptions=(ConnectionError,),
    )

    @retry_async(config)
    async def fails_with_connection_error():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("timeout")

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RetryExhaustedError):
            await fails_with_connection_error()

    assert call_count == 3


# ── Backoff behaviour ─────────────────────────────────────────────────────────


async def test_exponential_backoff_without_jitter():
    config = RetryConfig(
        max_attempts=4,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=100.0,
        jitter=False,
    )
    sleep_calls: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    @retry_async(config)
    async def always_fails():
        raise ValueError("fail")

    with patch("agentproof.core.retry.asyncio.sleep", side_effect=capture_sleep):
        with pytest.raises(RetryExhaustedError):
            await always_fails()

    # attempt 0 → sleep base*2^0=1.0, attempt 1 → 2.0, attempt 2 → 4.0
    assert sleep_calls == pytest.approx([1.0, 2.0, 4.0])


async def test_max_delay_caps_backoff():
    config = RetryConfig(
        max_attempts=5,
        base_delay=10.0,
        exponential_base=10.0,
        max_delay=15.0,
        jitter=False,
    )
    sleep_calls: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    @retry_async(config)
    async def always_fails():
        raise ValueError("fail")

    with patch("agentproof.core.retry.asyncio.sleep", side_effect=capture_sleep):
        with pytest.raises(RetryExhaustedError):
            await always_fails()

    assert all(d <= 15.0 for d in sleep_calls)


# ── Default config ────────────────────────────────────────────────────────────


def test_default_retry_config_values():
    config = RetryConfig()
    assert config.max_attempts == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True
    assert Exception in config.retryable_exceptions
