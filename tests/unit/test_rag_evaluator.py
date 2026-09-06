"""Unit tests for RAGEvaluator. All DeepEval and Qdrant calls are mocked."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.base import ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError
from agentproof.core.runner import TestRunner
from agentproof.evaluators.llm import LLMEvaluator
from agentproof.evaluators.rag import RAGEvaluator
from agentproof.integrations.qdrant import QdrantEvaluator, RetrievedPoint


QUERY = "What is our refund policy?"
EXPECTED = "Customers may request a refund within 30 days of purchase."
OUTPUT = "Refunds are available within 30 days of purchase."
DOCS = ["Our policy allows refunds within 30 days of purchase date."]


def _stub_metric(score: float, reason: str = "mocked reason"):
    instance = MagicMock()
    instance.score = score
    instance.reason = reason
    instance.success = score >= 0.5
    instance.a_measure = AsyncMock(return_value=score)
    cls = MagicMock(return_value=instance)
    return cls, instance


def _llm_result(metric: str, score: float, *, passed: bool | None = None) -> ValidationResult:
    from datetime import datetime, timezone
    import uuid

    return ValidationResult(
        passed=score >= 0.7 if passed is None else passed,
        score=score,
        evaluator_name="LLMEvaluator",
        metric=metric,
        threshold=0.7 if metric == "relevance" else 0.9,
        details={"reason": "delegated"},
        latency_ms=1.0,
        timestamp=datetime.now(timezone.utc),
        audit_id=str(uuid.uuid4()),
    )


@pytest.fixture
def config() -> AgentProofConfig:
    return AgentProofConfig()


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(config, mock_audit) -> RAGEvaluator:
    return RAGEvaluator(config, mock_audit)


@pytest.fixture(autouse=True)
def stub_llm_test_case(monkeypatch):
    monkeypatch.setattr(
        "agentproof.evaluators.rag.LLMTestCase",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


# ── Identity ──────────────────────────────────────────────────────────────────


def test_name(evaluator):
    assert evaluator.name == "RAGEvaluator"


# ── Contextual recall ─────────────────────────────────────────────────────────


async def test_contextual_recall_pass(evaluator):
    cls, _ = _stub_metric(0.85, "covers the gold answer")
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        result = await evaluator.evaluate_contextual_recall(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )

    assert result.passed is True
    assert result.score == 0.85
    assert result.metric == "contextual_recall"
    assert result.threshold == 0.7
    assert result.evaluator_name == "RAGEvaluator"
    assert result.details["reason"] == "covers the gold answer"


async def test_contextual_recall_fail(evaluator):
    cls, _ = _stub_metric(0.4)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        result = await evaluator.evaluate_contextual_recall(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    assert result.passed is False
    assert result.error is None


async def test_contextual_recall_passes_expected_output_and_docs():
    captured = {}

    def capture_case(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    cls, _ = _stub_metric(0.9)
    evaluator = RAGEvaluator(AgentProofConfig(), MagicMock(spec=AuditLogger))
    evaluator._audit.log = AsyncMock()

    with (
        patch("agentproof.evaluators.rag.LLMTestCase", side_effect=capture_case),
        patch("agentproof.evaluators.rag.ContextualRecallMetric", cls),
    ):
        await evaluator.evaluate_contextual_recall(
            query=QUERY,
            retrieved_docs=DOCS,
            expected_output=EXPECTED,
            actual_output=OUTPUT,
        )

    assert captured["input"] == QUERY
    assert captured["expected_output"] == EXPECTED
    assert captured["retrieval_context"] == DOCS
    assert captured["actual_output"] == OUTPUT


async def test_contextual_recall_empty_docs_is_quality_failure(evaluator, mock_audit):
    result = await evaluator.evaluate_contextual_recall(
        query=QUERY, retrieved_docs=[], expected_output=EXPECTED
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.error is None
    assert result.details["empty_retrieval"] is True
    mock_audit.log.assert_awaited_once()


async def test_contextual_recall_whitespace_docs_is_quality_failure(evaluator):
    result = await evaluator.evaluate_contextual_recall(
        query=QUERY, retrieved_docs=["  ", ""], expected_output=EXPECTED
    )
    assert result.details["empty_retrieval"] is True


async def test_contextual_recall_missing_expected_output_is_error(evaluator, mock_audit):
    result = await evaluator.evaluate_contextual_recall(
        query=QUERY, retrieved_docs=DOCS, expected_output="  "
    )
    assert result.passed is False
    assert result.error is not None
    assert "expected_output" in result.error
    mock_audit.log.assert_awaited_once()


async def test_contextual_recall_uses_threshold_and_judge(evaluator, config):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        await evaluator.evaluate_contextual_recall(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    kwargs = cls.call_args.kwargs
    assert kwargs["threshold"] == config.contextual_recall_threshold
    assert kwargs["model"] == config.judge_model
    assert kwargs["include_reason"] is True


# ── Contextual precision ──────────────────────────────────────────────────────


async def test_contextual_precision_pass(evaluator):
    cls, _ = _stub_metric(0.88)
    with patch("agentproof.evaluators.rag.ContextualPrecisionMetric", cls):
        result = await evaluator.evaluate_contextual_precision(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    assert result.passed is True
    assert result.metric == "contextual_precision"
    assert result.threshold == 0.8


async def test_contextual_precision_fail_below_threshold(evaluator):
    cls, _ = _stub_metric(0.75)
    with patch("agentproof.evaluators.rag.ContextualPrecisionMetric", cls):
        result = await evaluator.evaluate_contextual_precision(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    assert result.passed is False
    assert result.score == 0.75


async def test_contextual_precision_empty_docs_is_quality_failure(evaluator):
    result = await evaluator.evaluate_contextual_precision(
        query=QUERY, retrieved_docs=None, expected_output=EXPECTED
    )
    assert result.passed is False
    assert result.error is None
    assert result.metric == "contextual_precision"


# ── evaluate_retrieval dispatch ───────────────────────────────────────────────


async def test_evaluate_retrieval_defaults_to_recall(evaluator):
    cls, _ = _stub_metric(0.82)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        result = await evaluator.evaluate_retrieval(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    assert result.metric == "contextual_recall"


async def test_evaluate_retrieval_joins_ground_truth_docs(evaluator):
    captured = {}

    def capture_case(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    cls, _ = _stub_metric(0.9)
    with (
        patch("agentproof.evaluators.rag.LLMTestCase", side_effect=capture_case),
        patch("agentproof.evaluators.rag.ContextualRecallMetric", cls),
    ):
        await evaluator.evaluate_retrieval(
            query=QUERY,
            retrieved_docs=DOCS,
            ground_truth_docs=["Gold sentence one.", "Gold sentence two."],
        )
    assert captured["expected_output"] == "Gold sentence one.\nGold sentence two."


# ── Generation (delegates to LLMEvaluator) ────────────────────────────────────


async def test_generation_faithfulness_delegates(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_faithfulness = AsyncMock(
        return_value=_llm_result("faithfulness", 0.94, passed=True)
    )
    evaluator = RAGEvaluator(config, mock_audit, llm=llm)
    result = await evaluator.evaluate_generation_faithfulness(
        query=QUERY, actual_output=OUTPUT, retrieved_docs=DOCS, executive=True
    )
    llm.evaluate_faithfulness.assert_awaited_once()
    kwargs = llm.evaluate_faithfulness.call_args.kwargs
    assert kwargs["input"] == QUERY
    assert kwargs["output"] == OUTPUT
    assert kwargs["retrieval_context"] == DOCS
    assert kwargs["executive"] is True
    assert result.evaluator_name == "LLMEvaluator"
    assert result.score == 0.94


async def test_generation_relevance_delegates(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_relevance = AsyncMock(return_value=_llm_result("relevance", 0.81))
    evaluator = RAGEvaluator(config, mock_audit, llm=llm)
    result = await evaluator.evaluate_generation_relevance(
        query=QUERY, actual_output=OUTPUT
    )
    llm.evaluate_relevance.assert_awaited_once_with(input=QUERY, output=OUTPUT)
    assert result.metric == "relevance"


async def test_evaluate_generation_is_faithfulness(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_faithfulness = AsyncMock(
        return_value=_llm_result("faithfulness", 0.91, passed=True)
    )
    evaluator = RAGEvaluator(config, mock_audit, llm=llm)
    result = await evaluator.evaluate_generation(
        query=QUERY, actual_output=OUTPUT, retrieved_docs=DOCS
    )
    assert result.metric == "faithfulness"


# ── evaluate() dispatch ───────────────────────────────────────────────────────


async def test_evaluate_query_alias(evaluator):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        result = await evaluator.evaluate(
            query=QUERY,
            context=DOCS,
            expected_output=EXPECTED,
        )
    assert result.metric == "contextual_recall"


async def test_evaluate_dispatches_precision(evaluator):
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.rag.ContextualPrecisionMetric", cls):
        result = await evaluator.evaluate(
            input=QUERY,
            retrieval_context=DOCS,
            expected_output=EXPECTED,
            metric="precision",
        )
    assert result.metric == "contextual_precision"


async def test_evaluate_unknown_metric(evaluator, mock_audit):
    result = await evaluator.evaluate(query=QUERY, metric="toxicity")
    assert result.passed is False
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── evaluate_all / full pipeline ──────────────────────────────────────────────


async def test_evaluate_all_returns_four_results(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_faithfulness = AsyncMock(
        return_value=_llm_result("faithfulness", 0.93, passed=True)
    )
    llm.evaluate_relevance = AsyncMock(return_value=_llm_result("relevance", 0.84))
    evaluator = RAGEvaluator(config, mock_audit, llm=llm)

    recall, _ = _stub_metric(0.8)
    precision, _ = _stub_metric(0.85)
    with (
        patch("agentproof.evaluators.rag.ContextualRecallMetric", recall),
        patch("agentproof.evaluators.rag.ContextualPrecisionMetric", precision),
    ):
        results = await evaluator.evaluate_all(
            query=QUERY,
            retrieved_docs=DOCS,
            expected_output=EXPECTED,
            actual_output=OUTPUT,
        )

    assert [r.metric for r in results] == [
        "contextual_recall",
        "contextual_precision",
        "faithfulness",
        "relevance",
    ]


async def test_full_pipeline_without_qdrant_uses_provided_docs(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_faithfulness = AsyncMock(
        return_value=_llm_result("faithfulness", 0.93, passed=True)
    )
    llm.evaluate_relevance = AsyncMock(return_value=_llm_result("relevance", 0.84))
    evaluator = RAGEvaluator(config, mock_audit, llm=llm)
    recall, _ = _stub_metric(0.8)
    precision, _ = _stub_metric(0.85)
    with (
        patch("agentproof.evaluators.rag.ContextualRecallMetric", recall),
        patch("agentproof.evaluators.rag.ContextualPrecisionMetric", precision),
    ):
        results = await evaluator.evaluate_full_pipeline(
            query=QUERY,
            expected_output=EXPECTED,
            actual_output=OUTPUT,
            retrieved_docs=DOCS,
        )
    assert len(results) == 4


async def test_full_pipeline_missing_docs_and_qdrant_is_error(evaluator, mock_audit):
    results = await evaluator.evaluate_full_pipeline(
        query=QUERY, expected_output=EXPECTED, actual_output=OUTPUT
    )
    assert len(results) == 1
    assert results[0].error is not None
    assert "retrieved_docs" in results[0].error
    mock_audit.log.assert_awaited_once()


async def test_full_pipeline_searches_qdrant_and_scores_ids(config, mock_audit):
    llm = MagicMock(spec=LLMEvaluator)
    llm.evaluate_faithfulness = AsyncMock(
        return_value=_llm_result("faithfulness", 0.93, passed=True)
    )
    llm.evaluate_relevance = AsyncMock(return_value=_llm_result("relevance", 0.84))

    qdrant = MagicMock(spec=QdrantEvaluator)
    qdrant.integration.search = AsyncMock(
        return_value=[RetrievedPoint(id="a", score=0.9, payload={"text": DOCS[0]})]
    )
    qdrant.evaluate_retrieval_quality = AsyncMock(
        return_value=_llm_result("retrieval_quality", 1.0, passed=True)
    )

    evaluator = RAGEvaluator(config, mock_audit, llm=llm, qdrant=qdrant)
    recall, _ = _stub_metric(0.8)
    precision, _ = _stub_metric(0.85)
    with (
        patch("agentproof.evaluators.rag.ContextualRecallMetric", recall),
        patch("agentproof.evaluators.rag.ContextualPrecisionMetric", precision),
    ):
        results = await evaluator.evaluate_full_pipeline(
            query=QUERY,
            expected_output=EXPECTED,
            actual_output=OUTPUT,
            collection="policies",
            query_vector=[0.1, 0.2],
            ground_truth_ids=["a"],
            top_k=5,
        )

    qdrant.integration.search.assert_awaited_once()
    qdrant.evaluate_retrieval_quality.assert_awaited_once()
    assert results[0].metric == "retrieval_quality"
    assert len(results) == 5


# ── Errors / audit ────────────────────────────────────────────────────────────


async def test_deepeval_exception_becomes_error_result(evaluator, mock_audit):
    cls, inst = _stub_metric(0.9)
    inst.a_measure = AsyncMock(side_effect=RuntimeError("judge timed out"))
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await evaluator.evaluate_contextual_recall(
                query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
            )
    assert result.passed is False
    assert "judge timed out" in (result.error or "")
    mock_audit.log.assert_awaited_once()


async def test_audit_error_propagates(evaluator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    cls, _ = _stub_metric(0.9)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        with pytest.raises(AuditError, match="disk full"):
            await evaluator.evaluate_contextual_recall(
                query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
            )


async def test_score_clamped(evaluator):
    cls, _ = _stub_metric(1.4)
    with patch("agentproof.evaluators.rag.ContextualRecallMetric", cls):
        result = await evaluator.evaluate_contextual_recall(
            query=QUERY, retrieved_docs=DOCS, expected_output=EXPECTED
        )
    assert result.score == 1.0


# ── TestRunner ────────────────────────────────────────────────────────────────


async def test_runner_registers_rag_metrics(config, tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    evaluator = RAGEvaluator(config, runner.audit_logger)

    recall, _ = _stub_metric(0.85)
    precision, _ = _stub_metric(0.9)
    with (
        patch("agentproof.evaluators.rag.ContextualRecallMetric", recall),
        patch("agentproof.evaluators.rag.ContextualPrecisionMetric", precision),
    ):
        runner.register(
            evaluator,
            query=QUERY,
            retrieved_docs=DOCS,
            expected_output=EXPECTED,
            metric="contextual_recall",
        )
        runner.register(
            evaluator,
            query=QUERY,
            retrieved_docs=DOCS,
            expected_output=EXPECTED,
            metric="contextual_precision",
        )
        summary = await runner.run(parallel=False)

    assert summary.total == 2
    assert summary.passed == 2
    assert summary.all_passed is True
