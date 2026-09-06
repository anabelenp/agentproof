"""Async LiteLLM wrapper for multi-model completions and failover.

Isolates all LiteLLM usage so RoutingValidator can mock `litellm.acompletion`.
Does not require API keys at construction — live calls pick keys from config
or the process environment (LiteLLM's usual lookup).

Usage:
    client = LiteLLMIntegration(config)
    result = await client.complete("What is our refund policy?", model="claude-sonnet-4-6")
    print(result.text, result.model)
"""

import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import IntegrationError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async

DEFAULT_MAX_TOKENS = 1024

_LOW_COST_MARKERS = ("mini", "haiku", "nano", "flash", "lite")
_HIGH_COST_MARKERS = ("opus", "gpt-4", "o1", "o3")


@dataclass
class CompletionResult:
    """Normalized LiteLLM completion.

    Args:
        text: Assistant message content.
        model: Model name reported on the response (the model that served).
        requested_model: Model name passed into the call (router alias or id).
        latency_ms: Wall-clock time for this completion (includes failover).
        used_fallback: True when the primary model failed and fallback served.
        primary_error: Primary-model error message when fallback was used.
    """

    text: str
    model: str
    requested_model: str
    latency_ms: float
    used_fallback: bool = False
    primary_error: str | None = None


class LiteLLMIntegration:
    """Thin async wrapper around `litellm.acompletion`.

    Args:
        config: AgentProofConfig. API keys are optional at init; they are
            forwarded per-call when present so LiteLLM does not need env vars.
    """

    def __init__(self, config: AgentProofConfig) -> None:
        self.config = config
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                APIConnectionError,
                InternalServerError,
                RateLimitError,
                ServiceUnavailableError,
                Timeout,
                ConnectionError,
                TimeoutError,
            ),
        )

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        fallback_model: str | None = None,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> CompletionResult:
        """Complete a prompt, optionally failing over to `fallback_model`.

        When `fallback_model` is set the primary is tried once (no retry) so
        failover SLA measurements are not inflated by backoff. The fallback
        call is retried using the framework retry config.

        Args:
            prompt: User message. Must be non-empty.
            model: Primary / requested model (or router alias).
            fallback_model: Model to use if the primary call fails.
            system: Optional system prompt.
            max_tokens: Maximum tokens for the completion.

        Returns:
            CompletionResult with text, serving model, and failover flags.

        Raises:
            IntegrationError: Empty prompt/model, empty response, or both
                primary and fallback failed.
        """
        if not prompt or not prompt.strip():
            raise IntegrationError("prompt must be a non-empty string")
        if not model or not model.strip():
            raise IntegrationError("model must be a non-empty string")

        if fallback_model:
            started = time.perf_counter()
            try:
                result = await self._complete_once(
                    prompt, model, system=system, max_tokens=max_tokens
                )
                result.used_fallback = False
                return result
            except Exception as exc:
                result = await self._complete_with_retry(
                    prompt, fallback_model, system=system, max_tokens=max_tokens
                )
                result.used_fallback = True
                result.requested_model = model
                result.primary_error = f"{type(exc).__name__}: {exc}"
                result.latency_ms = (time.perf_counter() - started) * 1000.0
                return result

        return await self._complete_with_retry(
            prompt, model, system=system, max_tokens=max_tokens
        )

    async def _complete_with_retry(
        self,
        prompt: str,
        model: str,
        *,
        system: str | None,
        max_tokens: int,
    ) -> CompletionResult:
        """Retry `_complete_once` on transient LiteLLM failures.

        Args:
            prompt: User message.
            model: Model name.
            system: Optional system prompt.
            max_tokens: Maximum tokens.

        Returns:
            CompletionResult from the first successful attempt.

        Raises:
            IntegrationError: After retries are exhausted, or a non-retryable error.
        """

        async def _call() -> CompletionResult:
            return await self._complete_once(
                prompt, model, system=system, max_tokens=max_tokens
            )

        try:
            return await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"LiteLLM completion failed after {self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"LiteLLM completion failed: {exc}") from exc

    async def _complete_once(
        self,
        prompt: str,
        model: str,
        *,
        system: str | None,
        max_tokens: int,
    ) -> CompletionResult:
        """Single `litellm.acompletion` call with no retry.

        Args:
            prompt: User message.
            model: Model name.
            system: Optional system prompt.
            max_tokens: Maximum tokens.

        Returns:
            CompletionResult parsed from the LiteLLM response.

        Raises:
            IntegrationError: Response has no text content.
        """
        started = time.perf_counter()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        api_key = self._api_key_for(model)
        if api_key:
            kwargs["api_key"] = api_key

        response = await litellm.acompletion(**kwargs)
        text = extract_text(response)
        served = extract_model(response, default=model)
        return CompletionResult(
            text=text,
            model=served,
            requested_model=model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _api_key_for(self, model: str) -> str | None:
        """Select an API key from config based on the model name.

        Args:
            model: Requested model or router alias.

        Returns:
            Matching key, or None so LiteLLM can fall back to env vars.
        """
        name = model.lower()
        if "claude" in name or "anthropic" in name:
            return self.config.anthropic_api_key
        if any(tag in name for tag in ("gpt", "openai", "o1", "o3")):
            return self.config.openai_api_key
        return self.config.openai_api_key or self.config.anthropic_api_key


def extract_text(response: Any) -> str:
    """Pull assistant text from a LiteLLM ModelResponse or dict.

    Args:
        response: LiteLLM completion object (or a duck-typed mock).

    Returns:
        Non-empty assistant content string.

    Raises:
        IntegrationError: If no text content is present.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise IntegrationError("LiteLLM response contained no choices")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str) or not content:
        raise IntegrationError("LiteLLM response contained no text content")
    return content


def extract_model(response: Any, default: str = "") -> str:
    """Read the serving model name from a LiteLLM response.

    Args:
        response: LiteLLM completion object.
        default: Value used when the response does not report a model.

    Returns:
        Model name string.
    """
    model = getattr(response, "model", None)
    if not model and isinstance(response, dict):
        model = response.get("model")
    hidden = getattr(response, "_hidden_params", None)
    if not model and isinstance(hidden, dict):
        model = hidden.get("model") or hidden.get("model_id")
    return str(model or default)


def normalize_model_name(name: str) -> str:
    """Lowercase a model id and strip a provider prefix (`anthropic/claude-...`).

    Args:
        name: Raw model or alias string.

    Returns:
        Normalized model name.
    """
    normalized = name.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized


def models_match(actual: str, expected: str) -> bool:
    """True if two model names refer to the same model.

    Provider prefixes are ignored. A version suffix is allowed (`gpt-4` matches
    `gpt-4-turbo`) but `gpt-4` does not match `gpt-4o`.

    Args:
        actual: Model reported by the provider.
        expected: Model the test expected.

    Returns:
        True on a match.
    """
    actual_n = normalize_model_name(actual)
    expected_n = normalize_model_name(expected)
    if not actual_n or not expected_n:
        return False
    if actual_n == expected_n:
        return True
    return actual_n.startswith(expected_n + "-") or expected_n.startswith(actual_n + "-")


def is_high_cost_model(model: str) -> bool:
    """Heuristic: frontier/high-cost model vs small/cheap variants.

    Args:
        model: Model name or alias.

    Returns:
        True for opus / gpt-4 (non-mini) / o1 / o3 class models.
    """
    name = normalize_model_name(model)
    if any(tag in name for tag in _LOW_COST_MARKERS):
        return False
    return any(tag in name for tag in _HIGH_COST_MARKERS)
