"""Unit tests for LiteLLMIntegration and RoutingValidator. All LLM calls mocked."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.base import ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError
from agentproof.core.runner import TestRunner
from agentproof.evaluators.llm import LLMEvaluator
from agentproof.evaluators.routing import RoutingValidator
from agentproof.integrations.litellm import (
    CompletionResult,
    LiteLLMIntegration,
    extract_model,
    extract_text,
    is_high_cost_model,
    models_match,
    normalize_model_name,
)

PROMPT = "Summarize this insurance claim in one sentence."


def _response(text: str = "ok", model: str = "claude-sonnet-4-6") -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp._hidden_params = {}
    return resp


def _completion(
    *,
    text: str = "ok",
    model: str = "claude-sonnet-4-6",
    requested: str | None = None,
    latency_ms: float = 50.0,
    used_fallback: bool = False,
    primary_error: str | None = None,
) -> CompletionResult:
    return CompletionResult(
        text=text,
        model=model,
        requested_model=requested or model,
        latency_ms=latency_ms,
        used_fallback=used_fallback,
        primary_error=primary_error,
    )


def _llm_result(score: float, *, passed: bool | None = None, error: str | None = None) -> ValidationResult:
    return ValidationResult(
        passed=(score >= 0.7) if passed is None else passed,
        score=score,
        evaluator_name="LLMEvaluator",
        metric="relevance",
        threshold=0.7,
        details={},
        latency_ms=1.0,
        timestamp=datetime.now(timezone.utc),
        audit_id=str(uuid4()),
        error=error,
    )


@pytest.fixture
def mock_acompletion():
    with patch("agentproof.integrations.litellm.litellm.acompletion", new_callable=AsyncMock) as mocked:
        mocked.return_value = _response()
        yield mocked


@pytest.fixture
def integration(mock_acompletion) -> LiteLLMIntegration:
    return LiteLLMIntegration(AgentProofConfig())


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def litellm_mock() -> LiteLLMIntegration:
    client = MagicMock(spec=LiteLLMIntegration)
    client.complete = AsyncMock(return_value=_completion())
    return client


@pytest.fixture
def validator(mock_audit, litellm_mock) -> RoutingValidator:
    return RoutingValidator(AgentProofConfig(), mock_audit, litellm=litellm_mock)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_normalize_strips_provider_prefix():
    assert normalize_model_name("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_models_match_ignores_prefix():
    assert models_match("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6") is True


def test_models_match_allows_version_suffix():
    assert models_match("gpt-4-turbo", "gpt-4") is True


def test_models_match_does_not_confuse_gpt4_and_gpt4o():
    assert models_match("gpt-4o", "gpt-4") is False
    assert models_match("gpt-4o-mini", "gpt-4") is False


def test_models_match_empty_is_false():
    assert models_match("", "claude-sonnet-4-6") is False


def test_is_high_cost_opus_and_gpt4():
    assert is_high_cost_model("claude-opus-4") is True
    assert is_high_cost_model("gpt-4o") is True
    assert is_high_cost_model("o1-preview") is True


def test_is_high_cost_excludes_mini_and_haiku():
    assert is_high_cost_model("gpt-4o-mini") is False
    assert is_high_cost_model("claude-haiku-3-5") is False


def test_extract_text_from_object():
    assert extract_text(_response("hello")) == "hello"


def test_extract_text_from_dict():
    assert extract_text({"choices": [{"message": {"content": "dict-text"}}]}) == "dict-text"


def test_extract_text_empty_choices_raises():
    resp = MagicMock()
    resp.choices = []
    with pytest.raises(IntegrationError, match="no choices"):
        extract_text(resp)


def test_extract_model_falls_back_to_hidden_params():
    resp = MagicMock(spec=["_hidden_params"])
    resp._hidden_params = {"model": "hidden-model"}
    assert extract_model(resp, default="x") == "hidden-model"


# ── LiteLLMIntegration ────────────────────────────────────────────────────────


async def test_complete_returns_text_and_model(integration, mock_acompletion):
    mock_acompletion.return_value = _response("refunds in 30 days", "claude-sonnet-4-6")
    result = await integration.complete(PROMPT, model="claude-sonnet-4-6")
    assert result.text == "refunds in 30 days"
    assert result.model == "claude-sonnet-4-6"
    assert result.used_fallback is False
    assert result.requested_model == "claude-sonnet-4-6"


async def test_complete_sends_user_message(integration, mock_acompletion):
    await integration.complete(PROMPT, model="gpt-4o-mini")
    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": PROMPT}]
    assert kwargs["max_tokens"] == 1024


async def test_complete_forwards_system_prompt(integration, mock_acompletion):
    await integration.complete(PROMPT, model="gpt-4o-mini", system="Be brief.")
    messages = mock_acompletion.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Be brief."}


async def test_complete_forwards_anthropic_key(mock_acompletion):
    client = LiteLLMIntegration(AgentProofConfig(anthropic_api_key="sk-ant-test"))
    await client.complete(PROMPT, model="claude-sonnet-4-6")
    assert mock_acompletion.call_args.kwargs["api_key"] == "sk-ant-test"


async def test_complete_empty_prompt_raises(integration):
    with pytest.raises(IntegrationError, match="prompt"):
        await integration.complete("  ", model="gpt-4o-mini")


async def test_complete_retries_then_succeeds(integration, mock_acompletion):
    mock_acompletion.side_effect = [
        ConnectionError("blip"),
        _response("recovered", "gpt-4o-mini"),
    ]
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await integration.complete(PROMPT, model="gpt-4o-mini")
    assert result.text == "recovered"
    assert mock_acompletion.await_count == 2


async def test_complete_exhausted_retries_become_integration_error(integration, mock_acompletion):
    mock_acompletion.side_effect = ConnectionError("down")
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.complete(PROMPT, model="gpt-4o-mini")


async def test_complete_fallback_on_primary_failure(integration, mock_acompletion):
    mock_acompletion.side_effect = [
        RuntimeError("primary down"),
        _response("from fallback", "gpt-4o-mini"),
    ]
    result = await integration.complete(
        PROMPT,
        model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
    )
    assert result.used_fallback is True
    assert result.model == "gpt-4o-mini"
    assert result.requested_model == "claude-sonnet-4-6"
    assert result.primary_error is not None
    assert "primary down" in result.primary_error
    assert mock_acompletion.await_count == 2


async def test_complete_does_not_fallback_when_primary_succeeds(integration, mock_acompletion):
    mock_acompletion.return_value = _response("primary ok", "claude-sonnet-4-6")
    result = await integration.complete(
        PROMPT,
        model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
    )
    assert result.used_fallback is False
    assert mock_acompletion.await_count == 1


async def test_complete_empty_content_raises(integration, mock_acompletion):
    mock_acompletion.return_value = _response("", "claude-sonnet-4-6")
    with pytest.raises(IntegrationError, match="no text content"):
        await integration.complete(PROMPT, model="claude-sonnet-4-6")


# ── RoutingValidator identity / dispatch ──────────────────────────────────────


def test_name(validator):
    assert validator.name == "RoutingValidator"


async def test_unknown_metric_is_error_result(validator, mock_audit):
    result = await validator.evaluate(prompt=PROMPT, metric="latency")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── routing correctness ───────────────────────────────────────────────────────


async def test_routing_correctness_pass(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="claude-sonnet-4-6")
    result = await validator.validate_routing_correctness(
        prompt=PROMPT, expected_model="claude-sonnet-4-6"
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.metric == "routing_correctness"
    assert result.details["matched"] is True
    assert result.error is None


async def test_routing_correctness_detects_silent_misroute(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="gpt-4o")
    result = await validator.validate_routing_correctness(
        prompt=PROMPT, expected_model="claude-sonnet-4-6"
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["actual_model"] == "gpt-4o"
    assert result.error is None


async def test_routing_correctness_matches_provider_prefix(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="anthropic/claude-sonnet-4-6")
    result = await validator.validate_routing_correctness(
        prompt=PROMPT, expected_model="claude-sonnet-4-6"
    )
    assert result.passed is True


async def test_routing_correctness_empty_prompt_is_error(validator, mock_audit):
    result = await validator.validate_routing_correctness(
        prompt="  ", expected_model="claude-sonnet-4-6"
    )
    assert result.error is not None
    assert "prompt" in result.error
    mock_audit.log.assert_awaited_once()


async def test_routing_correctness_writes_audit(validator, mock_audit):
    result = await validator.validate_routing_correctness(
        prompt=PROMPT, expected_model="claude-sonnet-4-6"
    )
    mock_audit.log.assert_awaited_once()
    entry = mock_audit.log.call_args[0][0]
    assert entry.audit_id == result.audit_id
    assert entry.evaluator == "RoutingValidator"


async def test_evaluate_dispatches_routing_correctness(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="gpt-4o-mini")
    result = await validator.evaluate(
        input=PROMPT,
        expected_model="gpt-4o-mini",
        metric="routing_correctness",
    )
    assert result.metric == "routing_correctness"
    assert result.passed is True


# ── fallback ──────────────────────────────────────────────────────────────────


async def test_fallback_pass_within_sla(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(
        model="gpt-4o-mini",
        requested="claude-sonnet-4-6",
        latency_ms=800.0,
        used_fallback=True,
        primary_error="RateLimitError: overloaded",
    )
    result = await validator.validate_fallback(
        prompt=PROMPT,
        primary_model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
    )
    assert result.passed is True
    assert result.details["fallback_activated"] is True
    assert result.details["within_sla"] is True
    assert result.details["failover_seconds"] == 0.8


async def test_fallback_fails_when_primary_succeeds(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(
        model="claude-sonnet-4-6",
        used_fallback=False,
    )
    result = await validator.validate_fallback(
        prompt=PROMPT,
        primary_model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
    )
    assert result.passed is False
    assert result.details["fallback_activated"] is False
    assert result.error is None


async def test_fallback_fails_when_over_sla(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(
        model="gpt-4o-mini",
        latency_ms=8000.0,
        used_fallback=True,
    )
    result = await validator.validate_fallback(
        prompt=PROMPT,
        primary_model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
    )
    assert result.passed is False
    assert result.details["within_sla"] is False
    assert result.details["sla_seconds"] == 5.0


async def test_evaluate_dispatches_fallback(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(
        model="gpt-4o-mini", used_fallback=True, latency_ms=100.0
    )
    result = await validator.evaluate(
        prompt=PROMPT,
        primary_model="claude-sonnet-4-6",
        fallback_model="gpt-4o-mini",
        metric="fallback",
    )
    assert result.metric == "fallback"
    assert result.passed is True


# ── output consistency ────────────────────────────────────────────────────────


async def test_output_consistency_pass(config, mock_audit, litellm_mock):
    litellm_mock.complete.return_value = _completion(text="same idea")
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate = AsyncMock(return_value=_llm_result(0.88))
    validator = RoutingValidator(config, mock_audit, litellm=litellm_mock, llm=llm)

    result = await validator.validate_output_consistency(
        prompt=PROMPT, models=["claude-sonnet-4-6", "gpt-4o-mini"]
    )
    assert result.passed is True
    assert result.metric == "output_consistency"
    assert result.details["delta"] == 0.0
    assert llm.evaluate.await_count == 2


async def test_output_consistency_fails_on_large_delta(config, mock_audit, litellm_mock):
    litellm_mock.complete.return_value = _completion()
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate = AsyncMock(side_effect=[_llm_result(0.95), _llm_result(0.72)])
    validator = RoutingValidator(config, mock_audit, litellm=litellm_mock, llm=llm)

    result = await validator.validate_output_consistency(
        prompt=PROMPT, models=["claude-sonnet-4-6", "gpt-4o-mini"]
    )
    assert result.passed is False
    assert abs(result.details["delta"] - 0.23) < 1e-9
    assert result.threshold == 0.15


async def test_output_consistency_fails_when_one_model_below_quality(
    config, mock_audit, litellm_mock
):
    litellm_mock.complete.return_value = _completion()
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate = AsyncMock(
        side_effect=[
            _llm_result(0.91, passed=True),
            _llm_result(0.40, passed=False),
        ]
    )
    validator = RoutingValidator(config, mock_audit, litellm=litellm_mock, llm=llm)
    result = await validator.validate_output_consistency(
        prompt=PROMPT, models=["claude-sonnet-4-6", "gpt-4o-mini"]
    )
    assert result.passed is False
    assert "gpt-4o-mini" in result.details["quality_failures"]


async def test_output_consistency_requires_two_models(validator, mock_audit):
    result = await validator.validate_output_consistency(prompt=PROMPT, models=["only-one"])
    assert result.error is not None
    assert "at least two models" in result.error
    mock_audit.log.assert_awaited_once()


async def test_output_consistency_llm_error_is_error_result(config, mock_audit, litellm_mock):
    litellm_mock.complete.return_value = _completion()
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate = AsyncMock(return_value=_llm_result(0.0, passed=False, error="timeout"))
    validator = RoutingValidator(config, mock_audit, litellm=litellm_mock, llm=llm)
    result = await validator.validate_output_consistency(
        prompt=PROMPT, models=["a", "b"]
    )
    assert result.error is not None
    assert "timeout" in result.error


# ── cost routing ──────────────────────────────────────────────────────────────


async def test_cost_routing_low_tier_rejects_opus(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="claude-opus-4")
    result = await validator.validate_cost_routing(
        prompt=PROMPT, model="router-cheap", expected_tier="low"
    )
    assert result.passed is False
    assert result.details["is_high_cost"] is True


async def test_cost_routing_low_tier_accepts_mini(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="gpt-4o-mini")
    result = await validator.validate_cost_routing(
        prompt=PROMPT, model="router-cheap", expected_tier="low"
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_cost_routing_forbidden_list(validator, litellm_mock):
    litellm_mock.complete.return_value = _completion(model="gpt-4o-mini")
    result = await validator.validate_cost_routing(
        prompt=PROMPT,
        model="router-cheap",
        expected_tier="low",
        forbidden_models=["gpt-4o-mini"],
    )
    assert result.passed is False
    assert result.details["forbidden_hit"] is True


# ── evaluate_all / TestRunner / errors ────────────────────────────────────────


async def test_evaluate_all_returns_three_results(config, mock_audit, litellm_mock):
    litellm_mock.complete.return_value = _completion(
        model="claude-sonnet-4-6", used_fallback=True, latency_ms=100.0
    )
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate = AsyncMock(return_value=_llm_result(0.9))
    validator = RoutingValidator(config, mock_audit, litellm=litellm_mock, llm=llm)
    results = await validator.evaluate_all(
        prompt=PROMPT,
        expected_model="claude-sonnet-4-6",
        primary_model="claude-sonnet-4-6",
        fallback_model="claude-sonnet-4-6",
        models=["claude-sonnet-4-6", "gpt-4o-mini"],
    )
    assert [r.metric for r in results] == [
        "routing_correctness",
        "fallback",
        "output_consistency",
    ]


async def test_audit_error_propagates(validator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError, match="disk full"):
        await validator.validate_routing_correctness(
            prompt=PROMPT, expected_model="claude-sonnet-4-6"
        )


async def test_integration_failure_becomes_error_result(validator, litellm_mock, mock_audit):
    litellm_mock.complete.side_effect = IntegrationError("provider down")
    result = await validator.validate_routing_correctness(
        prompt=PROMPT, expected_model="claude-sonnet-4-6"
    )
    assert result.passed is False
    assert "provider down" in (result.error or "")
    mock_audit.log.assert_awaited_once()


async def test_runner_registers_routing_metrics(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    litellm_mock = MagicMock(spec=LiteLLMIntegration)
    litellm_mock.complete = AsyncMock(return_value=_completion(model="gpt-4o-mini"))
    validator = RoutingValidator(config, runner.audit_logger, litellm=litellm_mock)

    runner.register(
        validator,
        prompt=PROMPT,
        expected_model="gpt-4o-mini",
        metric="routing_correctness",
    )
    runner.register(
        validator,
        prompt=PROMPT,
        model="router-cheap",
        metric="cost_routing",
        expected_tier="low",
    )
    summary = await runner.run(parallel=False)
    assert summary.total == 2
    assert summary.passed == 2
    assert summary.all_passed is True
