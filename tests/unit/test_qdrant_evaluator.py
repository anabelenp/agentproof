"""Unit tests for QdrantIntegration and QdrantEvaluator. All Qdrant calls mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError
from agentproof.integrations.qdrant import (
    QdrantEvaluator,
    QdrantIntegration,
    RetrievedPoint,
    payload_text,
    precision_recall_at_k,
)


VECTOR = [0.1, 0.2, 0.3]


def _hit(point_id: str, score: float = 0.9, text: str = "doc") -> MagicMock:
    point = MagicMock()
    point.id = point_id
    point.score = score
    point.payload = {"text": text}
    return point


def _query_response(*hits: MagicMock) -> MagicMock:
    response = MagicMock()
    response.points = list(hits)
    return response


@pytest.fixture
def mock_client():
    with patch("agentproof.integrations.qdrant.QdrantClient") as cls:
        client = MagicMock()
        cls.return_value = client
        client.query_points.return_value = _query_response(_hit("doc-1"))
        count_result = MagicMock()
        count_result.count = 3
        client.count.return_value = count_result
        client.retrieve.return_value = [_hit("doc-1")]
        client.close = MagicMock()
        yield cls


@pytest.fixture
def integration(mock_client) -> QdrantIntegration:
    return QdrantIntegration(AgentProofConfig())


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def evaluator(mock_audit, integration) -> QdrantEvaluator:
    return QdrantEvaluator(AgentProofConfig(), mock_audit, integration=integration)


# ── precision_recall_at_k ─────────────────────────────────────────────────────


def test_precision_recall_perfect():
    precision, recall = precision_recall_at_k(["a", "b"], ["a", "b"], k=2)
    assert precision == 1.0
    assert recall == 1.0


def test_precision_recall_partial():
    precision, recall = precision_recall_at_k(["a", "x", "y"], ["a", "b"], k=2)
    # top-2 = a, x → 1 relevant / 2 = 0.5 precision; 1 / 2 truth = 0.5 recall
    assert precision == 0.5
    assert recall == 0.5


def test_precision_recall_empty_retrieval():
    precision, recall = precision_recall_at_k([], ["a", "b"], k=5)
    assert precision == 0.0
    assert recall == 0.0


def test_precision_recall_divides_precision_by_k():
    precision, recall = precision_recall_at_k(["a"], ["a", "b", "c"], k=5)
    assert precision == 1 / 5
    assert recall == 1 / 3


def test_precision_recall_k_zero():
    precision, recall = precision_recall_at_k(["a"], ["a"], k=0)
    assert precision == 0.0
    assert recall == 0.0


def test_precision_recall_normalizes_ids_to_str():
    precision, recall = precision_recall_at_k([1, 2], ["1"], k=2)
    assert precision == 0.5
    assert recall == 1.0


# ── payload_text ──────────────────────────────────────────────────────────────


def test_payload_text_prefers_text_key():
    assert payload_text({"text": "hello", "content": "other"}) == "hello"


def test_payload_text_falls_back_to_content():
    assert payload_text({"content": "body"}) == "body"


def test_payload_text_empty_payload():
    assert payload_text(None) == ""
    assert payload_text({}) == ""


# ── QdrantIntegration ─────────────────────────────────────────────────────────


def test_init_passes_url_and_skips_compat_check(mock_client):
    QdrantIntegration(AgentProofConfig(qdrant_url="http://qdrant:6333"))
    mock_client.assert_called_once()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["url"] == "http://qdrant:6333"
    assert kwargs["check_compatibility"] is False
    assert "api_key" not in kwargs


def test_init_passes_api_key_when_set(mock_client):
    QdrantIntegration(AgentProofConfig(qdrant_api_key="q-secret"))
    assert mock_client.call_args.kwargs["api_key"] == "q-secret"


async def test_search_returns_retrieved_points(integration, mock_client):
    mock_client.return_value.query_points.return_value = _query_response(
        _hit("a", 0.95, "alpha"),
        _hit("b", 0.80, "beta"),
    )
    hits = await integration.search("policies", VECTOR, top_k=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].text == "alpha"
    assert hits[0].score == 0.95


async def test_search_forwards_collection_vector_and_limit(integration, mock_client):
    await integration.search("policies", VECTOR, top_k=5, score_threshold=0.4)
    kwargs = mock_client.return_value.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "policies"
    assert kwargs["query"] == VECTOR
    assert kwargs["limit"] == 5
    assert kwargs["with_payload"] is True
    assert kwargs["score_threshold"] == 0.4


async def test_search_empty_collection_raises(integration):
    with pytest.raises(IntegrationError, match="collection"):
        await integration.search("  ", VECTOR)


async def test_search_empty_vector_raises(integration):
    with pytest.raises(IntegrationError, match="query_vector"):
        await integration.search("policies", [])


async def test_search_retries_connection_error(integration, mock_client):
    mock_client.return_value.query_points.side_effect = [
        ConnectionError("down"),
        _query_response(_hit("a")),
    ]
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        hits = await integration.search("policies", VECTOR)
    assert hits[0].id == "a"
    assert mock_client.return_value.query_points.call_count == 2


async def test_search_exhausted_retries_become_integration_error(integration, mock_client):
    mock_client.return_value.query_points.side_effect = ConnectionError("down")
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(IntegrationError, match="failed after"):
            await integration.search("policies", VECTOR)


async def test_count_returns_integer(integration, mock_client):
    result = MagicMock()
    result.count = 42
    mock_client.return_value.count.return_value = result
    assert await integration.count("policies") == 42


async def test_retrieve_returns_matching_points(integration, mock_client):
    mock_client.return_value.retrieve.return_value = [_hit("doc-9", text="kept")]
    hits = await integration.retrieve("policies", ["doc-9"])
    assert hits[0].id == "doc-9"
    assert hits[0].text == "kept"


async def test_retrieve_empty_ids_short_circuits(integration, mock_client):
    hits = await integration.retrieve("policies", [])
    assert hits == []
    mock_client.return_value.retrieve.assert_not_called()


async def test_close_delegates_to_client(integration, mock_client):
    await integration.close()
    mock_client.return_value.close.assert_called_once()


# ── QdrantEvaluator identity / dispatch ───────────────────────────────────────


def test_name(evaluator):
    assert evaluator.name == "QdrantEvaluator"


async def test_unknown_metric_is_error_result(evaluator, mock_audit):
    result = await evaluator.evaluate(metric="schema", collection="policies")
    assert result.passed is False
    assert result.error is not None
    assert "Unknown metric" in result.error
    mock_audit.log.assert_awaited_once()


# ── retrieval quality ─────────────────────────────────────────────────────────


async def test_retrieval_quality_pass(evaluator, mock_client):
    mock_client.return_value.query_points.return_value = _query_response(
        _hit("a"), _hit("b"), _hit("c"), _hit("d"), _hit("e")
    )
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=["a", "b", "c", "d", "e"],
        top_k=5,
        query="refund policy",
    )
    assert result.passed is True
    assert result.metric == "retrieval_quality"
    assert result.score == 1.0
    assert result.details["precision_at_k"] == 1.0
    assert result.details["recall_at_k"] == 1.0
    assert result.details["query"] == "refund policy"
    assert result.error is None


async def test_retrieval_quality_fail_low_precision(evaluator, mock_client):
    mock_client.return_value.query_points.return_value = _query_response(
        _hit("a"), _hit("x"), _hit("y"), _hit("z"), _hit("w")
    )
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=["a", "b"],
        top_k=5,
    )
    assert result.passed is False
    assert result.details["precision_at_k"] == 0.2
    assert result.details["missing_ids"] == ["b"]
    assert "x" in result.details["extra_ids"]
    assert result.error is None


async def test_retrieval_quality_empty_retrieval_is_quality_failure(
    evaluator, mock_client, mock_audit
):
    mock_client.return_value.query_points.return_value = _query_response()
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=["a"],
        top_k=5,
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.error is None
    assert result.details["empty_retrieval"] is True
    mock_audit.log.assert_awaited_once()


async def test_retrieval_quality_missing_ground_truth_is_error(evaluator, mock_audit):
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=[],
    )
    assert result.passed is False
    assert result.error is not None
    assert "ground_truth_ids" in result.error
    mock_audit.log.assert_awaited_once()


async def test_retrieval_quality_writes_audit(evaluator, mock_audit, mock_client):
    mock_client.return_value.query_points.return_value = _query_response(_hit("a"))
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=["a"],
        top_k=1,
    )
    mock_audit.log.assert_awaited_once()
    entry = mock_audit.log.call_args[0][0]
    assert entry.audit_id == result.audit_id
    assert entry.evaluator == "QdrantEvaluator"


async def test_evaluate_dispatches_retrieval_quality(evaluator, mock_client):
    mock_client.return_value.query_points.return_value = _query_response(
        _hit("a"), _hit("b")
    )
    result = await evaluator.evaluate(
        metric="retrieval_quality",
        collection="policies",
        query_vector=VECTOR,
        ground_truth_ids=["a", "b"],
        top_k=2,
    )
    assert result.metric == "retrieval_quality"
    assert result.passed is True


# ── data integrity ────────────────────────────────────────────────────────────


async def test_data_integrity_pass(evaluator, mock_client):
    count_result = MagicMock()
    count_result.count = 3
    mock_client.return_value.count.return_value = count_result
    mock_client.return_value.retrieve.return_value = [_hit("a"), _hit("b")]
    result = await evaluator.evaluate_data_integrity(
        collection="policies",
        expected_count=3,
        sample_ids=["a", "b"],
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["actual_count"] == 3
    assert result.details["missing_ids"] == []


async def test_data_integrity_count_mismatch(evaluator, mock_client):
    count_result = MagicMock()
    count_result.count = 9
    mock_client.return_value.count.return_value = count_result
    result = await evaluator.evaluate_data_integrity(
        collection="policies",
        expected_count=3,
        sample_ids=[],
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["actual_count"] == 9


async def test_data_integrity_missing_sample_ids(evaluator, mock_client):
    count_result = MagicMock()
    count_result.count = 2
    mock_client.return_value.count.return_value = count_result
    mock_client.return_value.retrieve.return_value = [_hit("a")]
    result = await evaluator.evaluate_data_integrity(
        collection="policies",
        expected_count=2,
        sample_ids=["a", "missing"],
    )
    assert result.passed is False
    assert result.details["missing_ids"] == ["missing"]
    assert result.details["found_ids"] == ["a"]
    # count ok + 1 of 2 samples → 2/3
    assert abs(result.score - 2 / 3) < 1e-9


# ── latency ───────────────────────────────────────────────────────────────────


async def test_retrieval_latency_under_sla(evaluator, mock_client):
    result = await evaluator.evaluate_retrieval_latency(
        collection="policies",
        query_vector=VECTOR,
        top_k=5,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.metric == "retrieval_latency"
    assert result.details["sla_ms"] == 500.0
    assert result.details["retrieval_latency_ms"] >= 0.0


async def test_retrieval_latency_over_sla(evaluator, mock_client):
    evaluator._elapsed_ms = lambda start: 900.0  # type: ignore[method-assign]
    result = await evaluator.evaluate_retrieval_latency(
        collection="policies",
        query_vector=VECTOR,
    )
    assert result.passed is False
    assert result.details["retrieval_latency_ms"] == 900.0
    assert abs(result.score - 500.0 / 900.0) < 1e-9


# ── errors ────────────────────────────────────────────────────────────────────


async def test_search_failure_becomes_error_result(evaluator, mock_client, mock_audit):
    mock_client.return_value.query_points.side_effect = RuntimeError("cluster down")
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await evaluator.evaluate_retrieval_quality(
            collection="policies",
            query_vector=VECTOR,
            ground_truth_ids=["a"],
        )
    assert result.passed is False
    assert result.error is not None
    assert "cluster down" in result.error
    mock_audit.log.assert_awaited_once()


async def test_audit_error_propagates(evaluator, mock_audit, mock_client):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    mock_client.return_value.query_points.return_value = _query_response(_hit("a"))
    with pytest.raises(AuditError, match="disk full"):
        await evaluator.evaluate_retrieval_quality(
            collection="policies",
            query_vector=VECTOR,
            ground_truth_ids=["a"],
            top_k=1,
        )


def test_retrieved_point_text_property():
    point = RetrievedPoint(id="1", score=0.5, payload={"page_content": "chunk"})
    assert point.text == "chunk"
