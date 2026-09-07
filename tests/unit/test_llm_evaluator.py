"""Unit tests for LLMEvaluator. All DeepEval and Anthropic calls are mocked."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.base import ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError
from agentproof.core.runner import TestRunner
from agentproof.evaluators.llm import LLMEvaluator
from agentproof.integrations.anthropic import AnthropicIntegration


QUERY = "What is our refund policy?"
OUTPUT = "Refunds are available within 30 days of purchase."
CONTEXT = ["Our policy allows refunds within 30 days of purchase date."]


def _stub_metric(score: float, reason: str = "mocked reason", success: bool | None = None):
    instance = MagicMock()
    instance.score = score
    instance.reason = reason
    instance.success = (score >= 0.5) if success is None else success
    instance.a_measure = AsyncMock(return_value=score)
    cls = MagicMock(return_value=instance)
    return cls, instance


@pytest.fixture
def config() -> AgentProofConfig:
    return AgentProofConfig()


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(config, mock_audit) -> LLMEvaluator:
    return LLMEvaluator(config, mock_audit)


@pytest.fixture(autouse=True)
def stub_llm_test_case(monkeypatch):
    """Avoid exercising DeepEval's LLMTestCase constructor in unit tests."""
    monkeypatch.setattr(
        "agentproof.evaluators.llm.LLMTestCase",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


# ── Identity ──────────────────────────────────────────────────────────────────


def test_name(evaluator):
    assert evaluator.name == "LLMEvaluator"


# ── Relevance ─────────────────────────────────────────────────────────────────


async def test_relevance_pass(evaluator):
    cls, _ = _stub_metric(0.85, "answers the question")
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)

    assert result.passed is True
    assert result.score == 0.85
    assert result.metric == "relevance"
    assert result.threshold == 0.7
    assert result.error is None
    assert result.details["reason"] == "answers the question"
    assert result.details["score_interpretation"] == "higher is better"


async def test_relevance_fail_below_threshold(evaluator):
    cls, _ = _stub_metric(0.4, "off topic")
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)

    assert result.passed is False
    assert result.score == 0.4
    assert result.error is None


async def test_relevance_passes_threshold_and_model_to_deepeval(evaluator, config):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)

    kwargs = cls.call_args.kwargs
    assert kwargs["threshold"] == config.default_relevance_threshold
    assert kwargs["include_reason"] is True
    # No API key in unit tests — judge is the configured model name string.
    assert kwargs["model"] == config.judge_model


async def test_claude_judge_wrapped_when_api_key_present(mock_audit):
    evaluator = LLMEvaluator(
        AgentProofConfig(anthropic_api_key="sk-test"),
        mock_audit,
    )
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert cls.call_args.kwargs["model"].__class__.__name__ == "AnthropicModel"


async def test_non_claude_judge_passed_as_string(mock_audit):
    evaluator = LLMEvaluator(AgentProofConfig(judge_model="gpt-4o"), mock_audit)
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert cls.call_args.kwargs["model"] == "gpt-4o"


async def test_relevance_calls_a_measure(evaluator):
    cls, inst = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    inst.a_measure.assert_awaited_once()


async def test_relevance_writes_audit(evaluator, mock_audit):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)

    mock_audit.log.assert_awaited_once()
    entry = mock_audit.log.call_args[0][0]
    assert entry.audit_id == result.audit_id
    assert entry.evaluator == "LLMEvaluator"
    assert entry.metric == "relevance"
    assert entry.model == evaluator.config.judge_model


async def test_relevance_score_exactly_at_threshold_passes(evaluator):
    cls, _ = _stub_metric(0.7)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert result.passed is True


async def test_relevance_empty_input_is_error_result(evaluator, mock_audit):
    result = await evaluator.evaluate_relevance(input="  ", output=OUTPUT)
    assert result.passed is False
    assert result.error is not None
    assert "input" in result.error
    mock_audit.log.assert_awaited_once()


async def test_relevance_empty_output_is_error_result(evaluator):
    result = await evaluator.evaluate_relevance(input=QUERY, output="")
    assert result.passed is False
    assert "output" in (result.error or "")


# ── Faithfulness ──────────────────────────────────────────────────────────────


async def test_faithfulness_pass(evaluator):
    cls, _ = _stub_metric(0.94, "grounded in policy docs")
    with patch("agentproof.evaluators.llm.FaithfulnessMetric", cls):
        result = await evaluator.evaluate_faithfulness(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert result.passed is True
    assert result.score == 0.94
    assert result.metric == "faithfulness"
    assert result.threshold == 0.9


async def test_faithfulness_fail(evaluator):
    cls, _ = _stub_metric(0.62, "contradicts source")
    with patch("agentproof.evaluators.llm.FaithfulnessMetric", cls):
        result = await evaluator.evaluate_faithfulness(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )
    assert result.passed is False
    assert result.error is None


async def test_faithfulness_uses_retrieval_context_field():
    captured = {}

    def capture_case(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    cls, _ = _stub_metric(0.95)
    evaluator = LLMEvaluator(AgentProofConfig(), MagicMock(spec=AuditLogger))
    evaluator._audit.log = AsyncMock()

    with (
        patch("agentproof.evaluators.llm.LLMTestCase", side_effect=capture_case),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", cls),
    ):
        await evaluator.evaluate_faithfulness(
            input=QUERY, output=OUTPUT, retrieval_context=CONTEXT
        )

    assert captured["retrieval_context"] == CONTEXT
    assert captured["actual_output"] == OUTPUT


async def test_faithfulness_prefers_retrieval_context_over_context():
    captured = {}

    def capture_case(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    cls, _ = _stub_metric(0.95)
    evaluator = LLMEvaluator(AgentProofConfig(), MagicMock(spec=AuditLogger))
    evaluator._audit.log = AsyncMock()

    with (
        patch("agentproof.evaluators.llm.LLMTestCase", side_effect=capture_case),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", cls),
    ):
        await evaluator.evaluate_faithfulness(
            input=QUERY,
            output=OUTPUT,
            context=["fallback"],
            retrieval_context=CONTEXT,
        )

    assert captured["retrieval_context"] == CONTEXT


async def test_faithfulness_missing_context_is_error_result(evaluator, mock_audit):
    result = await evaluator.evaluate_faithfulness(input=QUERY, output=OUTPUT)
    assert result.passed is False
    assert result.score == 0.0
    assert result.error is not None
    assert "context" in result.error
    mock_audit.log.assert_awaited_once()


async def test_faithfulness_empty_context_is_error_result(evaluator):
    result = await evaluator.evaluate_faithfulness(
        input=QUERY, output=OUTPUT, context=[]
    )
    assert result.error is not None
    assert "context" in result.error


async def test_faithfulness_executive_threshold(evaluator):
    cls, _ = _stub_metric(0.92)
    with patch("agentproof.evaluators.llm.FaithfulnessMetric", cls):
        result = await evaluator.evaluate_faithfulness(
            input=QUERY, output=OUTPUT, context=CONTEXT, executive=True
        )

    assert result.threshold == 0.95
    assert result.passed is False  # 0.92 < 0.95
    assert cls.call_args.kwargs["threshold"] == 0.95


async def test_faithfulness_executive_pass(evaluator):
    cls, _ = _stub_metric(0.97)
    with patch("agentproof.evaluators.llm.FaithfulnessMetric", cls):
        result = await evaluator.evaluate_faithfulness(
            input=QUERY, output=OUTPUT, context=CONTEXT, executive=True
        )
    assert result.passed is True


# ── Hallucination ─────────────────────────────────────────────────────────────


async def test_hallucination_converts_alignment_score_to_rate(evaluator):
    # DeepEval 4.x: 0.92 alignment → AgentProof rate 0.08
    cls, _ = _stub_metric(0.92, "mostly grounded")
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        result = await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert result.metric == "hallucination"
    assert result.threshold == 0.1
    assert abs(result.score - 0.08) < 1e-9
    assert result.passed is True  # 0.08 <= 0.1
    assert result.details["deepeval_score"] == 0.92
    assert "hallucination_rate" in result.details["score_interpretation"]


async def test_hallucination_fail_when_rate_above_threshold(evaluator):
    # 0.70 alignment → 0.30 hallucination rate > 0.1
    cls, _ = _stub_metric(0.70)
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        result = await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert abs(result.score - 0.30) < 1e-9
    assert result.passed is False
    assert result.error is None


async def test_hallucination_perfect_alignment_is_zero_rate(evaluator):
    cls, _ = _stub_metric(1.0)
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        result = await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )
    assert result.score == 0.0
    assert result.passed is True


async def test_hallucination_passes_inverted_threshold_to_deepeval(evaluator):
    cls, _ = _stub_metric(0.95)
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )
    # AgentProof rate threshold 0.1 → DeepEval min alignment 0.9
    assert abs(cls.call_args.kwargs["threshold"] - 0.9) < 1e-9


async def test_hallucination_uses_context_field_not_retrieval_context():
    captured = {}

    def capture_case(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    cls, _ = _stub_metric(0.95)
    evaluator = LLMEvaluator(AgentProofConfig(), MagicMock(spec=AuditLogger))
    evaluator._audit.log = AsyncMock()

    with (
        patch("agentproof.evaluators.llm.LLMTestCase", side_effect=capture_case),
        patch("agentproof.evaluators.llm.HallucinationMetric", cls),
    ):
        await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert captured["context"] == CONTEXT
    assert "retrieval_context" not in captured


async def test_hallucination_missing_context_is_error_result(evaluator, mock_audit):
    result = await evaluator.evaluate_hallucination(input=QUERY, output=OUTPUT)
    assert result.passed is False
    assert result.error is not None
    assert "context" in result.error
    mock_audit.log.assert_awaited_once()


async def test_hallucination_rate_exactly_at_threshold_passes(evaluator):
    # alignment 0.9 → rate 0.1 == threshold
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        result = await evaluator.evaluate_hallucination(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )
    assert result.passed is True
    assert abs(result.score - 0.1) < 1e-9


# ── Error handling ────────────────────────────────────────────────────────────


async def test_deepeval_exception_becomes_error_result(evaluator, mock_audit):
    cls, inst = _stub_metric(0.9)
    inst.a_measure = AsyncMock(side_effect=RuntimeError("judge timed out"))
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)

    assert result.passed is False
    assert result.score == 0.0
    assert result.error is not None
    assert "judge timed out" in result.error
    assert "attempts" in result.error
    mock_audit.log.assert_awaited_once()


async def test_none_score_becomes_error_result(evaluator):
    cls, inst = _stub_metric(0.9)
    inst.score = None
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert result.error is not None
    assert "no score" in result.error


async def test_score_above_one_is_clamped(evaluator):
    cls, _ = _stub_metric(1.4)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert result.score == 1.0
    assert result.passed is True


async def test_score_below_zero_is_clamped(evaluator):
    cls, _ = _stub_metric(-0.2)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert result.score == 0.0
    assert result.passed is False


async def test_audit_error_propagates(evaluator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        with pytest.raises(AuditError, match="disk full"):
            await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)


async def test_audit_error_on_validation_failure_propagates(evaluator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError):
        await evaluator.evaluate_relevance(input=" ", output=OUTPUT)


# ── evaluate() dispatch ───────────────────────────────────────────────────────


async def test_evaluate_defaults_to_relevance(evaluator):
    cls, _ = _stub_metric(0.88)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate(input=QUERY, output=OUTPUT)
    assert result.metric == "relevance"
    assert result.score == 0.88


async def test_evaluate_answer_relevancy_alias(evaluator):
    cls, _ = _stub_metric(0.8)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate(
            input=QUERY, output=OUTPUT, metric="answer_relevancy"
        )
    assert result.metric == "relevance"


async def test_evaluate_dispatches_faithfulness(evaluator):
    cls, _ = _stub_metric(0.91)
    with patch("agentproof.evaluators.llm.FaithfulnessMetric", cls):
        result = await evaluator.evaluate(
            input=QUERY, output=OUTPUT, context=CONTEXT, metric="faithfulness"
        )
    assert result.metric == "faithfulness"


async def test_evaluate_dispatches_hallucination(evaluator):
    cls, _ = _stub_metric(0.95)
    with patch("agentproof.evaluators.llm.HallucinationMetric", cls):
        result = await evaluator.evaluate(
            input=QUERY, output=OUTPUT, context=CONTEXT, metric="hallucination"
        )
    assert result.metric == "hallucination"
    assert abs(result.score - 0.05) < 1e-9


async def test_toxicity_pass(evaluator):
    cls, _ = _stub_metric(0.02)
    with patch("agentproof.evaluators.llm.ToxicityMetric", cls):
        result = await evaluator.evaluate_toxicity(input=QUERY, output=OUTPUT)
    assert result.passed is True
    assert result.score == 0.02
    assert result.metric == "toxicity"
    assert result.threshold == 0.1
    assert result.details["score_interpretation"] == "lower is better"


async def test_toxicity_fail_above_threshold(evaluator):
    cls, _ = _stub_metric(0.4)
    with patch("agentproof.evaluators.llm.ToxicityMetric", cls):
        result = await evaluator.evaluate_toxicity(input=QUERY, output=OUTPUT)
    assert result.passed is False
    assert result.score == 0.4
    assert result.error is None


async def test_evaluate_dispatches_toxicity(evaluator):
    cls, _ = _stub_metric(0.0)
    with patch("agentproof.evaluators.llm.ToxicityMetric", cls):
        result = await evaluator.evaluate(input=QUERY, output=OUTPUT, metric="toxicity")
    assert result.metric == "toxicity"
    assert result.passed is True


async def test_evaluate_unknown_metric_is_error_result(evaluator, mock_audit):
    result = await evaluator.evaluate(input=QUERY, output=OUTPUT, metric="bleu")
    assert result.passed is False
    assert result.error is not None
    assert "Unknown metric" in result.error
    mock_audit.log.assert_awaited_once()


async def test_evaluate_missing_output_without_anthropic_is_error(evaluator, mock_audit):
    result = await evaluator.evaluate(input=QUERY)
    assert result.passed is False
    assert result.error is not None
    assert "AnthropicIntegration" in result.error
    mock_audit.log.assert_awaited_once()


async def test_evaluate_generates_output_via_anthropic(config, mock_audit):
    anthropic = MagicMock(spec=AnthropicIntegration)
    anthropic.complete = AsyncMock(return_value=OUTPUT)
    evaluator = LLMEvaluator(config, mock_audit, anthropic=anthropic)

    cls, _ = _stub_metric(0.86)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate(input=QUERY)

    anthropic.complete.assert_awaited_once_with(QUERY)
    assert result.score == 0.86
    assert result.error is None


# ── evaluate_all ──────────────────────────────────────────────────────────────


async def test_evaluate_all_returns_four_results(evaluator):
    rel, _ = _stub_metric(0.85)
    faith, _ = _stub_metric(0.94)
    hall, _ = _stub_metric(0.96)
    tox, _ = _stub_metric(0.02)

    with (
        patch("agentproof.evaluators.llm.AnswerRelevancyMetric", rel),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", faith),
        patch("agentproof.evaluators.llm.HallucinationMetric", hall),
        patch("agentproof.evaluators.llm.ToxicityMetric", tox),
    ):
        results = await evaluator.evaluate_all(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert len(results) == 4
    assert [r.metric for r in results] == [
        "relevance",
        "faithfulness",
        "hallucination",
        "toxicity",
    ]
    assert all(isinstance(r, ValidationResult) for r in results)


async def test_evaluate_all_generates_output_once(config, mock_audit):
    anthropic = MagicMock(spec=AnthropicIntegration)
    anthropic.complete = AsyncMock(return_value=OUTPUT)
    evaluator = LLMEvaluator(config, mock_audit, anthropic=anthropic)

    rel, _ = _stub_metric(0.9)
    faith, _ = _stub_metric(0.95)
    hall, _ = _stub_metric(0.99)
    tox, _ = _stub_metric(0.01)

    with (
        patch("agentproof.evaluators.llm.AnswerRelevancyMetric", rel),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", faith),
        patch("agentproof.evaluators.llm.HallucinationMetric", hall),
        patch("agentproof.evaluators.llm.ToxicityMetric", tox),
    ):
        await evaluator.evaluate_all(input=QUERY, context=CONTEXT)

    assert anthropic.complete.await_count == 1


async def test_evaluate_all_continues_after_one_metric_errors(evaluator, mock_audit):
    rel, rel_inst = _stub_metric(0.9)
    rel_inst.a_measure = AsyncMock(side_effect=RuntimeError("boom"))
    faith, _ = _stub_metric(0.95)
    hall, _ = _stub_metric(0.99)
    tox, _ = _stub_metric(0.01)

    with (
        patch("agentproof.evaluators.llm.AnswerRelevancyMetric", rel),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", faith),
        patch("agentproof.evaluators.llm.HallucinationMetric", hall),
        patch("agentproof.evaluators.llm.ToxicityMetric", tox),
        patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock),
    ):
        results = await evaluator.evaluate_all(
            input=QUERY, output=OUTPUT, context=CONTEXT
        )

    assert results[0].error is not None
    assert results[1].error is None
    assert results[2].error is None
    assert results[3].error is None
    assert mock_audit.log.await_count == 4


# ── Result shape ──────────────────────────────────────────────────────────────


async def test_result_has_valid_audit_id(evaluator):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric", cls):
        result = await evaluator.evaluate_relevance(input=QUERY, output=OUTPUT)
    assert result.audit_id
    assert result.latency_ms >= 0.0
    assert result.timestamp.tzinfo is not None
    assert result.evaluator_name == "LLMEvaluator"


# ── TestRunner integration ────────────────────────────────────────────────────


async def test_runner_registers_three_metrics(config, tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    evaluator = LLMEvaluator(config, runner.audit_logger)

    rel, _ = _stub_metric(0.85)
    faith, _ = _stub_metric(0.94)
    hall, _ = _stub_metric(0.97)

    with (
        patch("agentproof.evaluators.llm.AnswerRelevancyMetric", rel),
        patch("agentproof.evaluators.llm.FaithfulnessMetric", faith),
        patch("agentproof.evaluators.llm.HallucinationMetric", hall),
    ):
        runner.register(evaluator, input=QUERY, output=OUTPUT, metric="relevance")
        runner.register(
            evaluator,
            input=QUERY,
            output=OUTPUT,
            context=CONTEXT,
            metric="faithfulness",
        )
        runner.register(
            evaluator,
            input=QUERY,
            output=OUTPUT,
            context=CONTEXT,
            metric="hallucination",
        )
        summary = await runner.run(parallel=False)

    assert summary.total == 3
    assert summary.passed == 3
    assert summary.all_passed is True
    assert summary.error_count == 0
