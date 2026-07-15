"""Exponential backoff retry decorator for async functions.

All LLM and external API calls in AgentProof must be wrapped with retry_async
to handle transient failures (rate limits, timeouts, 5xx errors).

Usage:
    from agentproof.core.retry import RetryConfig, retry_async

    config = RetryConfig(max_attempts=5, base_delay=0.5, jitter=True)

    @retry_async(config)
    async def call_llm(prompt: str) -> str:
        return await anthropic_client.messages.create(...)
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from .errors import RetryExhaustedError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic.

    Args:
        max_attempts: Total number of attempts (including the first).
        base_delay: Initial sleep duration in seconds before the second attempt.
        max_delay: Upper bound on sleep duration regardless of exponent.
        exponential_base: Multiplier applied per attempt (delay = base * base^attempt).
        jitter: If True, multiply delay by a random factor in [0.5, 1.5].
        retryable_exceptions: Only retry when one of these exception types is raised.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )


def retry_async(
    config: RetryConfig | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that retries an async function with exponential backoff.

    Args:
        config: RetryConfig instance. Uses defaults (3 attempts, 1s base) if None.

    Returns:
        Decorator that wraps the async function with retry logic.

    Raises:
        RetryExhaustedError: After all attempts fail, carrying the last exception.

    Example:
        @retry_async(RetryConfig(max_attempts=5))
        async def fetch_from_api(...) -> dict: ...
    """
    _config = config or RetryConfig()

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: Exception | None = None

            for attempt in range(_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except _config.retryable_exceptions as exc:
                    last_exc = exc

                    if attempt == _config.max_attempts - 1:
                        break

                    delay = min(
                        _config.base_delay * (_config.exponential_base**attempt),
                        _config.max_delay,
                    )
                    if _config.jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.2fs",
                        attempt + 1,
                        _config.max_attempts,
                        func.__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

            raise RetryExhaustedError(
                f"{func.__name__!r} failed after {_config.max_attempts} attempts",
                last_exception=last_exc,
            )

        return wrapper  # type: ignore[return-value]

    return decorator
