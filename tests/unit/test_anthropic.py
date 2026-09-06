"""Unit tests for AnthropicIntegration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIConnectionError, RateLimitError

from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import ConfigurationError, IntegrationError
from agentproof.integrations.anthropic import AnthropicIntegration, extract_text


def _config(**overrides) -> AgentProofConfig:
    defaults = dict(anthropic_api_key="sk-test-key")
    defaults.update(overrides)
    return AgentProofConfig(**defaults)


def _message(*texts: str, extra_blocks: list | None = None) -> MagicMock:
    blocks = []
    for text in texts:
        block = MagicMock()
        block.text = text
        blocks.append(block)
    if extra_blocks:
        blocks.extend(extra_blocks)
    message = MagicMock()
    message.content = blocks
    return message


@pytest.fixture
def mock_async_anthropic():
    with patch("agentproof.integrations.anthropic.AsyncAnthropic") as cls:
        client = AsyncMock()
        cls.return_value = client
        client.messages.create = AsyncMock(return_value=_message("hello from claude"))
        client.close = AsyncMock()
        yield cls


@pytest.fixture
def integration(mock_async_anthropic) -> AnthropicIntegration:
    return AnthropicIntegration(_config())


# ── Construction ──────────────────────────────────────────────────────────────


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicIntegration(_config(anthropic_api_key=None))


def test_empty_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicIntegration(_config(anthropic_api_key=""))


def test_init_passes_api_key_to_sdk(mock_async_anthropic):
    AnthropicIntegration(_config(anthropic_api_key="sk-live"))
    mock_async_anthropic.assert_called_once_with(api_key="sk-live")


# ── complete() ────────────────────────────────────────────────────────────────


async def test_complete_returns_text(integration, mock_async_anthropic):
    text = await integration.complete("What is our refund policy?")
    assert text == "hello from claude"


async def test_complete_sends_user_message(integration, mock_async_anthropic):
    await integration.complete("hello")
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


async def test_complete_uses_judge_model_by_default(integration, mock_async_anthropic):
    await integration.complete("hello")
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == integration.config.judge_model


async def test_complete_model_override(integration, mock_async_anthropic):
    await integration.complete("hello", model="claude-opus-4-7")
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"


async def test_complete_forwards_system_prompt(integration, mock_async_anthropic):
    await integration.complete("hello", system="You are a claims assistant.")
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["system"] == "You are a claims assistant."


async def test_complete_omits_system_when_not_set(integration, mock_async_anthropic):
    await integration.complete("hello")
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert "system" not in kwargs


async def test_complete_forwards_max_tokens(integration, mock_async_anthropic):
    await integration.complete("hello", max_tokens=256)
    kwargs = mock_async_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 256


async def test_complete_concatenates_multiple_text_blocks(integration, mock_async_anthropic):
    mock_async_anthropic.return_value.messages.create = AsyncMock(
        return_value=_message("Refunds ", "within 30 days.")
    )
    text = await integration.complete("policy?")
    assert text == "Refunds within 30 days."


async def test_complete_skips_non_text_blocks(integration, mock_async_anthropic):
    tool_block = MagicMock()
    tool_block.text = None
    mock_async_anthropic.return_value.messages.create = AsyncMock(
        return_value=_message("ok", extra_blocks=[tool_block])
    )
    text = await integration.complete("hi")
    assert text == "ok"


async def test_complete_empty_prompt_raises(integration):
    with pytest.raises(IntegrationError, match="non-empty"):
        await integration.complete("   ")


async def test_complete_empty_content_raises(integration, mock_async_anthropic):
    empty = MagicMock()
    empty.content = []
    mock_async_anthropic.return_value.messages.create = AsyncMock(return_value=empty)
    with pytest.raises(IntegrationError, match="no text content"):
        await integration.complete("hello")


# ── Retry ─────────────────────────────────────────────────────────────────────


async def test_complete_retries_rate_limit_then_succeeds(integration, mock_async_anthropic):
    create = AsyncMock(
        side_effect=[
            RateLimitError("slow down", response=MagicMock(), body=None),
            _message("recovered"),
        ]
    )
    mock_async_anthropic.return_value.messages.create = create

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        text = await integration.complete("hello")

    assert text == "recovered"
    assert create.call_count == 2


async def test_complete_retries_connection_error(integration, mock_async_anthropic):
    create = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),
            _message("ok"),
        ]
    )
    mock_async_anthropic.return_value.messages.create = create

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        text = await integration.complete("hello")

    assert text == "ok"


async def test_complete_exhausted_retries_become_integration_error(
    integration, mock_async_anthropic
):
    mock_async_anthropic.return_value.messages.create = AsyncMock(
        side_effect=RateLimitError("slow down", response=MagicMock(), body=None)
    )

    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.complete("hello")


async def test_complete_does_not_retry_value_error(integration, mock_async_anthropic):
    mock_async_anthropic.return_value.messages.create = AsyncMock(
        side_effect=ValueError("bad request shape")
    )
    with pytest.raises(IntegrationError, match="bad request shape"):
        await integration.complete("hello")
    assert mock_async_anthropic.return_value.messages.create.call_count == 1


# ── extract_text ──────────────────────────────────────────────────────────────


def test_extract_text_joins_blocks():
    assert extract_text(_message("a", "b", "c")) == "abc"


def test_extract_text_no_content_raises():
    message = MagicMock(spec=[])
    with pytest.raises(IntegrationError, match="no content"):
        extract_text(message)


def test_extract_text_ignores_empty_strings():
    blank = MagicMock()
    blank.text = ""
    message = _message("kept", extra_blocks=[blank])
    assert extract_text(message) == "kept"


# ── close ─────────────────────────────────────────────────────────────────────


async def test_close_delegates_to_client(integration, mock_async_anthropic):
    await integration.close()
    mock_async_anthropic.return_value.close.assert_awaited_once()
