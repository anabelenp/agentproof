"""Unit tests for StreamingValidator. HTTP is mocked; most tests use chunks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError
from agentproof.core.runner import TestRunner
from agentproof.evaluators.streaming import (
    StreamingValidator,
    count_tokens,
    parse_sse_fragment,
)


def _words(n: int) -> str:
    return " ".join(f"tok{i}" for i in range(n))


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def validator(mock_audit) -> StreamingValidator:
    return StreamingValidator(AgentProofConfig(), mock_audit)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_count_tokens():
    assert count_tokens("one two three") == 3
    assert count_tokens("   ") == 0


def test_parse_sse_data_and_done():
    text, done = parse_sse_fragment("data: Hello\ndata: [DONE]\n")
    assert text == "Hello"
    assert done is True


def test_parse_raw_text_without_sse():
    text, done = parse_sse_fragment("plain chunk")
    assert text == "plain chunk"
    assert done is False


# ── identity ──────────────────────────────────────────────────────────────────


def test_name(validator):
    assert validator.name == "StreamingValidator"


async def test_unknown_metric(validator, mock_audit):
    result = await validator.evaluate(metric="jitter", chunks=["x"])
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


async def test_missing_chunks_and_endpoint_is_error(validator):
    result = await validator.validate_ttft()
    assert result.error is not None
    assert "chunks or endpoint" in result.error


# ── TTFT ──────────────────────────────────────────────────────────────────────


async def test_ttft_pass_under_sla(validator):
    result = await validator.validate_ttft(
        chunks=["Hello"],
        t0=0.0,
        first_token_at=0.4,
        ended_at=1.0,
        done=True,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["ttft_seconds"] == 0.4
    assert result.details["sla_seconds"] == 2.0
    assert result.error is None


async def test_ttft_fail_over_sla(validator):
    result = await validator.validate_ttft(
        chunks=["Hello"],
        t0=0.0,
        first_token_at=3.0,
        ended_at=4.0,
        done=True,
    )
    assert result.passed is False
    assert abs(result.score - 2.0 / 3.0) < 1e-9
    assert result.error is None


async def test_ttft_no_token_fails(validator):
    result = await validator.validate_ttft(chunks=[], t0=0.0, ended_at=1.0, done=True)
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["ttft_seconds"] is None


async def test_ttft_error_before_token_is_error_result(validator):
    result = await validator.validate_ttft(
        chunks=[],
        t0=0.0,
        ended_at=0.5,
        error="ConnectError: refused",
        status_code=None,
    )
    assert result.passed is False
    assert result.error is not None
    assert "before first token" in result.error


# ── throughput ────────────────────────────────────────────────────────────────


async def test_throughput_pass(validator):
    # 40 tokens in 1 second → 40 tps >= 20
    result = await validator.validate_throughput(
        chunks=[_words(40)],
        t0=0.0,
        first_token_at=0.0,
        ended_at=1.0,
        done=True,
    )
    assert result.passed is True
    assert result.details["tokens_per_second"] == 40.0
    assert result.score == 1.0


async def test_throughput_fail_below_sla(validator):
    # 10 tokens in 1 second → 10 tps < 20
    result = await validator.validate_throughput(
        chunks=[_words(10)],
        t0=0.0,
        first_token_at=0.0,
        ended_at=1.0,
        done=True,
    )
    assert result.passed is False
    assert result.details["tokens_per_second"] == 10.0
    assert abs(result.score - 0.5) < 1e-9


async def test_throughput_stream_error_is_error_result(validator):
    result = await validator.validate_throughput(
        chunks=[_words(5)],
        t0=0.0,
        ended_at=1.0,
        error="ReadTimeout",
    )
    assert result.error is not None
    assert "ReadTimeout" in result.error


# ── completeness ──────────────────────────────────────────────────────────────


async def test_completeness_pass_with_done_marker(validator):
    result = await validator.validate_completeness(chunks=["Hello world", "data: [DONE]"])
    assert result.passed is True
    assert result.details["done"] is True
    assert result.details["has_text"] is True


async def test_completeness_fail_without_done(validator):
    result = await validator.validate_completeness(chunks=["partial answer"], done=False)
    assert result.passed is False
    assert result.details["done"] is False
    assert result.error is None


async def test_completeness_fail_empty_body(validator):
    result = await validator.validate_completeness(chunks=["[DONE]"], done=True)
    assert result.passed is False
    assert result.details["has_text"] is False


# ── graceful degradation ──────────────────────────────────────────────────────


async def test_graceful_degradation_visible_failure(validator):
    result = await validator.validate_graceful_degradation(mock_failure=True)
    assert result.passed is True
    assert result.details["visible_failure"] is True
    assert result.details["status_code"] == 503


async def test_graceful_degradation_silent_empty_200_fails(validator):
    result = await validator.validate_graceful_degradation(
        mock_failure=False,
        chunks=[],
        status_code=200,
        error=None,
        done=True,
    )
    assert result.passed is False
    assert result.details["silent_failure"] is True


# ── error handling ────────────────────────────────────────────────────────────


async def test_error_handling_pass_when_interruption_surfaced(validator):
    result = await validator.validate_error_handling(
        chunks=["partial "],
        t0=0.0,
        first_token_at=0.1,
        ended_at=0.4,
        done=False,
        error="ConnectionResetError: peer closed",
    )
    assert result.passed is True
    assert result.details["error_surfaced"] is True


async def test_error_handling_fail_silent_partial(validator):
    result = await validator.validate_error_handling(
        chunks=["partial answer without terminator"],
        done=False,
        error=None,
        status_code=200,
    )
    assert result.passed is False
    assert result.details["silent_partial"] is True


# ── evaluate_all / dispatch / httpx / audit ───────────────────────────────────


async def test_evaluate_all_three_metrics(validator, mock_audit):
    results = await validator.evaluate_all(
        chunks=[_words(40), "[DONE]"],
        t0=0.0,
        first_token_at=0.2,
        ended_at=1.0,
        done=True,
    )
    assert [r.metric for r in results] == ["ttft", "throughput", "completeness"]
    assert all(r.passed for r in results)
    assert mock_audit.log.await_count == 3


async def test_evaluate_dispatches_throughput(validator):
    result = await validator.evaluate(
        metric="throughput",
        chunks=[_words(40)],
        t0=0.0,
        ended_at=1.0,
        done=True,
    )
    assert result.metric == "throughput"
    assert result.passed is True


async def test_http_stream_uses_injected_client(mock_audit):
    async def _aiter():
        yield "data: Hello\n"
        yield "data: [DONE]\n"

    response = MagicMock()
    response.status_code = 200
    response.aiter_text = _aiter

    stream_cm = AsyncMock()
    stream_cm.__aenter__.return_value = response
    stream_cm.__aexit__.return_value = False

    client = MagicMock()
    client.stream.return_value = stream_cm
    client.aclose = AsyncMock()

    validator = StreamingValidator(AgentProofConfig(), mock_audit, client=client)
    result = await validator.validate_completeness(
        endpoint="http://llm.local/stream",
        payload={"prompt": "hi"},
    )
    client.stream.assert_called_once()
    assert result.passed is True
    assert result.details["has_text"] is True
    client.aclose.assert_not_awaited()  # injected client is not closed


async def test_http_error_status_recorded(mock_audit):
    async def _aiter():
        if False:
            yield ""

    response = MagicMock()
    response.status_code = 503
    response.aiter_text = _aiter
    stream_cm = AsyncMock()
    stream_cm.__aenter__.return_value = response
    stream_cm.__aexit__.return_value = False
    client = MagicMock()
    client.stream.return_value = stream_cm

    validator = StreamingValidator(AgentProofConfig(), mock_audit, client=client)
    result = await validator.validate_graceful_degradation(
        endpoint="http://llm.local/stream",
        payload={},
        mock_failure=False,
    )
    assert result.passed is True
    assert result.details["status_code"] == 503


async def test_audit_error_propagates(validator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError):
        await validator.validate_ttft(chunks=["x"], t0=0.0, first_token_at=0.1, ended_at=0.2)


async def test_runner_registers_streaming(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    validator = StreamingValidator(config, runner.audit_logger)
    runner.register(
        validator,
        metric="ttft",
        chunks=["Hello"],
        t0=0.0,
        first_token_at=0.3,
        ended_at=0.8,
        done=True,
    )
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1
