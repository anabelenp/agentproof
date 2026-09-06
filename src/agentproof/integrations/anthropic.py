"""Async Anthropic SDK wrapper.

Isolates all Anthropic SDK usage so evaluators can mock a single class.
Initializes `AsyncAnthropic` from config, constructs messages, calls
`client.messages.create(...)`, and returns parsed text content.

Usage:
    client = AnthropicIntegration(config)
    text = await client.complete("What is our refund policy?")
    await client.close()
"""

from typing import Any

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)

from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import ConfigurationError, IntegrationError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async

DEFAULT_MAX_TOKENS = 1024


class AnthropicIntegration:
    """Thin async wrapper around the official Anthropic SDK.

    All Anthropic HTTP traffic in AgentProof should go through this class.
    Callers receive plain strings; they never handle SDK response objects.

    Args:
        config: AgentProofConfig. `anthropic_api_key` is required.

    Raises:
        ConfigurationError: If `anthropic_api_key` is missing.
    """

    def __init__(self, config: AgentProofConfig) -> None:
        if not config.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is required")

        self.config = config
        self._client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            ),
        )

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Send a user prompt and return concatenated text blocks.

        Retries transient Anthropic failures (connection, timeout, rate limit,
        5xx) using the framework retry config. Non-retryable errors and
        exhausted retries surface as IntegrationError.

        Args:
            prompt: User message content. Must be non-empty.
            model: Override of `config.judge_model`.
            system: Optional system prompt.
            max_tokens: Maximum tokens for the completion.

        Returns:
            Concatenated text from response content blocks.

        Raises:
            IntegrationError: Empty prompt, empty response, or retries exhausted.
        """
        if not prompt or not prompt.strip():
            raise IntegrationError("prompt must be a non-empty string")

        resolved_model = model or self.config.judge_model

        async def _call() -> str:
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            message = await self._client.messages.create(**kwargs)
            return extract_text(message)

        try:
            return await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Anthropic completion failed after {self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Anthropic completion failed: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def extract_text(message: Any) -> str:
    """Pull concatenated text from an Anthropic Messages API response.

    Args:
        message: SDK `Message` object (or a duck-typed equivalent with
            a `content` sequence of blocks exposing optional `text`).

    Returns:
        Joined text of every block that has a non-empty `text` attribute.

    Raises:
        IntegrationError: If the response contains no text content.
    """
    content = getattr(message, "content", None)
    if content is None:
        raise IntegrationError("Anthropic response contained no content")

    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)

    if not parts:
        raise IntegrationError("Anthropic response contained no text content")

    return "".join(parts)
