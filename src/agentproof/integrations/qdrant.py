"""Qdrant vector-store integration and retrieval evaluator.

`QdrantIntegration` isolates the Qdrant SDK. `QdrantEvaluator` scores
retrieval quality (precision@k / recall@k), collection integrity, and
retrieval latency against configured SLAs.

Usage:
    qdrant = QdrantIntegration(config)
    hits = await qdrant.search("policies", query_vector=[0.1, 0.2], top_k=5)

    evaluator = QdrantEvaluator(config, audit_logger, integration=qdrant)
    result = await evaluator.evaluate_retrieval_quality(
        collection="policies",
        query_vector=[0.1, 0.2],
        ground_truth_ids=["doc-1", "doc-2"],
        top_k=5,
    )
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError, RAGEvaluatorError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async

DEFAULT_TOP_K = 5
PAYLOAD_TEXT_KEYS = ("text", "content", "page_content", "document", "chunk")

SUPPORTED_METRICS = (
    "retrieval_quality",
    "data_integrity",
    "retrieval_latency",
)


@dataclass
class RetrievedPoint:
    """A single Qdrant hit with a normalized string id.

    Args:
        id: Point id coerced to str (Qdrant ids may be int, UUID, or str).
        score: Similarity score returned by Qdrant.
        payload: Raw payload dict from the point.
    """

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Best-effort document text extracted from known payload keys."""
        return payload_text(self.payload)


def payload_text(payload: dict[str, Any] | None) -> str:
    """Extract document text from a Qdrant payload.

    Args:
        payload: Point payload, or None.

    Returns:
        First non-empty string among common text keys, else "".
    """
    if not payload:
        return ""
    for key in PAYLOAD_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def precision_recall_at_k(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    k: int,
) -> tuple[float, float]:
    """Compute precision@k and recall@k over id sets.

    Precision@k = (relevant in top k) / k.
    Recall@k    = (relevant in top k) / |ground truth|.

    Args:
        retrieved_ids: Ranked ids from the retriever (highest score first).
        ground_truth_ids: Ids that should have been retrieved.
        k: Cutoff. If k < 1 the scores are 0.0.

    Returns:
        (precision_at_k, recall_at_k), each in [0.0, 1.0].
    """
    if k < 1:
        return 0.0, 0.0
    truth = {str(item) for item in ground_truth_ids}
    top = [str(item) for item in retrieved_ids[:k]]
    hits = sum(1 for item in top if item in truth)
    precision = hits / k
    recall = hits / len(truth) if truth else 0.0
    return precision, recall


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall. 0.0 when both are 0."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class QdrantIntegration:
    """Thin async wrapper around the Qdrant client.

    Sync SDK calls run in `asyncio.to_thread` so evaluators stay async.
    A missing API key is allowed — local Qdrant does not require one.

    Args:
        config: AgentProofConfig. `qdrant_url` selects the server.
    """

    def __init__(self, config: AgentProofConfig) -> None:
        self.config = config
        kwargs: dict[str, Any] = {
            "url": config.qdrant_url,
            "check_compatibility": False,
        }
        if config.qdrant_api_key:
            kwargs["api_key"] = config.qdrant_api_key
        self._client = QdrantClient(**kwargs)
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                UnexpectedResponse,
                ResponseHandlingException,
            ),
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ) -> list[RetrievedPoint]:
        """Nearest-neighbor search over a collection.

        Args:
            collection: Qdrant collection name.
            query_vector: Query embedding.
            top_k: Maximum number of hits to return.
            score_threshold: Optional minimum similarity; forwarded to Qdrant.

        Returns:
            Ranked RetrievedPoint list (highest score first).

        Raises:
            IntegrationError: Invalid arguments, or retries exhausted.
        """
        if not collection or not collection.strip():
            raise IntegrationError("collection must be a non-empty string")
        if not query_vector:
            raise IntegrationError("query_vector must be a non-empty list of floats")
        if top_k < 1:
            raise IntegrationError("top_k must be >= 1")

        async def _call() -> Any:
            return await asyncio.to_thread(
                self._client.query_points,
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold,
            )

        try:
            response = await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Qdrant search failed after {self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Qdrant search failed: {exc}") from exc

        points = getattr(response, "points", None) or []
        return [_to_retrieved(point) for point in points]

    async def count(self, collection: str) -> int:
        """Return the number of points in a collection.

        Args:
            collection: Qdrant collection name.

        Returns:
            Point count.

        Raises:
            IntegrationError: Invalid collection name, or the call failed.
        """
        if not collection or not collection.strip():
            raise IntegrationError("collection must be a non-empty string")

        async def _call() -> Any:
            return await asyncio.to_thread(
                self._client.count,
                collection_name=collection,
                exact=True,
            )

        try:
            result = await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Qdrant count failed after {self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Qdrant count failed: {exc}") from exc

        return int(getattr(result, "count", 0))

    async def retrieve(self, collection: str, ids: list[str]) -> list[RetrievedPoint]:
        """Fetch points by id.

        Args:
            collection: Qdrant collection name.
            ids: Point ids to look up.

        Returns:
            RetrievedPoint list for ids that exist (missing ids are omitted).

        Raises:
            IntegrationError: Invalid arguments, or the call failed.
        """
        if not collection or not collection.strip():
            raise IntegrationError("collection must be a non-empty string")
        if not ids:
            return []

        async def _call() -> Any:
            return await asyncio.to_thread(
                self._client.retrieve,
                collection_name=collection,
                ids=list(ids),
                with_payload=True,
            )

        try:
            records = await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Qdrant retrieve failed after {self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Qdrant retrieve failed: {exc}") from exc

        return [_to_retrieved(record) for record in records or []]

    async def close(self) -> None:
        """Close the underlying Qdrant client if it exposes close()."""
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def _to_retrieved(point: Any) -> RetrievedPoint:
    """Normalize a Qdrant ScoredPoint or Record into RetrievedPoint.

    Args:
        point: SDK point object with `id` and optional `score` / `payload`.

    Returns:
        RetrievedPoint with string id.
    """
    payload = getattr(point, "payload", None) or {}
    score = getattr(point, "score", None)
    return RetrievedPoint(
        id=str(getattr(point, "id")),
        score=float(score) if score is not None else 0.0,
        payload=dict(payload) if isinstance(payload, dict) else {},
    )


class QdrantEvaluator(BaseEvaluator):
    """Scores Qdrant retrieval quality, integrity, and latency.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with thresholds and retrieval SLA.
        audit_logger: AuditLogger that receives every result.
        integration: Optional QdrantIntegration. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        integration: QdrantIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._integration = integration or QdrantIntegration(config)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "QdrantEvaluator"

    @property
    def integration(self) -> QdrantIntegration:
        """The QdrantIntegration used for search / count / retrieve."""
        return self._integration

    async def evaluate(
        self,
        *,
        metric: str = "retrieval_quality",
        collection: str = "",
        query_vector: list[float] | None = None,
        ground_truth_ids: list[str] | None = None,
        top_k: int = DEFAULT_TOP_K,
        expected_count: int | None = None,
        sample_ids: list[str] | None = None,
        query: str = "",
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a Qdrant evaluation method.

        Args:
            metric: One of "retrieval_quality" (default), "data_integrity",
                "retrieval_latency".
            collection: Target collection.
            query_vector: Query embedding for search-based metrics.
            ground_truth_ids: Relevant ids for retrieval_quality.
            top_k: Cutoff for search and precision@k / recall@k.
            expected_count: Expected point count for data_integrity.
            sample_ids: Ids that must exist for data_integrity.
            query: Optional query string stored in audit details.
            score_threshold: Optional Qdrant similarity floor.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        if key == "retrieval_quality":
            return await self.evaluate_retrieval_quality(
                collection=collection,
                query_vector=query_vector or [],
                ground_truth_ids=ground_truth_ids or [],
                top_k=top_k,
                query=query,
                score_threshold=score_threshold,
            )
        if key == "data_integrity":
            return await self.evaluate_data_integrity(
                collection=collection,
                expected_count=expected_count if expected_count is not None else 0,
                sample_ids=sample_ids or [],
            )
        if key == "retrieval_latency":
            return await self.evaluate_retrieval_latency(
                collection=collection,
                query_vector=query_vector or [],
                top_k=top_k,
                query=query,
            )

        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=metric,
            threshold=0.0,
            details={"supported_metrics": list(SUPPORTED_METRICS)},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
            error=(
                f"Unknown metric {metric!r}. "
                f"Expected one of: {', '.join(SUPPORTED_METRICS)}"
            ),
        )
        await self._write_audit(result)
        return result

    async def evaluate_retrieval_quality(
        self,
        *,
        collection: str,
        query_vector: list[float],
        ground_truth_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
        query: str = "",
        score_threshold: float | None = None,
    ) -> ValidationResult:
        """Score precision@k and recall@k against ground-truth ids.

        `passed` requires precision@k >= contextual_precision_threshold and
        recall@k >= contextual_recall_threshold. `score` is the F1 of the two.

        Empty retrieval is a quality failure (score 0, error unset), not an
        evaluator error.

        Args:
            collection: Qdrant collection name.
            query_vector: Query embedding.
            ground_truth_ids: Ids that should appear in the top-k hits.
            top_k: Search cutoff and k for precision/recall (default 5).
            query: Optional original query text, stored in details.
            score_threshold: Optional Qdrant similarity floor.

        Returns:
            ValidationResult. metric is "retrieval_quality".
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        threshold = min(
            self.config.contextual_precision_threshold,
            self.config.contextual_recall_threshold,
        )
        try:
            if not ground_truth_ids:
                raise RAGEvaluatorError(
                    "retrieval_quality requires non-empty ground_truth_ids",
                    evaluator_name=self.name,
                    metric="retrieval_quality",
                )
            hits = await self._integration.search(
                collection,
                query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            retrieved_ids = [hit.id for hit in hits]
            precision, recall = precision_recall_at_k(
                retrieved_ids, list(ground_truth_ids), top_k
            )
            truth = {str(item) for item in ground_truth_ids}
            retrieved_set = set(retrieved_ids)
            score = _clamp(_f1(precision, recall))
            passed = (
                precision >= self.config.contextual_precision_threshold
                and recall >= self.config.contextual_recall_threshold
            )
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="retrieval_quality",
                threshold=threshold,
                details={
                    "query": query,
                    "collection": collection,
                    "k": top_k,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "retrieved_ids": retrieved_ids,
                    "ground_truth_ids": [str(i) for i in ground_truth_ids],
                    "missing_ids": sorted(truth - retrieved_set),
                    "extra_ids": [i for i in retrieved_ids if i not in truth],
                    "empty_retrieval": len(hits) == 0,
                    "hit_scores": [hit.score for hit in hits],
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "retrieval_quality", exc, threshold=threshold
            )

        await self._write_audit(result)
        return result

    async def evaluate_data_integrity(
        self,
        *,
        collection: str,
        expected_count: int,
        sample_ids: list[str] | None = None,
    ) -> ValidationResult:
        """Confirm collection size and that sample ids exist.

        Args:
            collection: Qdrant collection name.
            expected_count: Expected number of points.
            sample_ids: Ids that must be retrievable. Optional.

        Returns:
            ValidationResult. score is the fraction of checks that passed
            (count match + each sample id). metric is "data_integrity".
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        samples = list(sample_ids or [])
        try:
            actual_count = await self._integration.count(collection)
            found: list[str] = []
            missing: list[str] = []
            if samples:
                records = await self._integration.retrieve(collection, samples)
                found_set = {record.id for record in records}
                for sid in samples:
                    key = str(sid)
                    if key in found_set:
                        found.append(key)
                    else:
                        missing.append(key)

            count_ok = actual_count == expected_count
            samples_ok = not missing
            checks = 1 + len(samples)
            passed_checks = (1 if count_ok else 0) + len(found)
            score = _clamp(passed_checks / checks) if checks else 1.0
            result = ValidationResult(
                passed=count_ok and samples_ok,
                score=score,
                evaluator_name=self.name,
                metric="data_integrity",
                threshold=1.0,
                details={
                    "collection": collection,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "sample_ids": [str(i) for i in samples],
                    "found_ids": found,
                    "missing_ids": missing,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "data_integrity", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_retrieval_latency(
        self,
        *,
        collection: str,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        query: str = "",
    ) -> ValidationResult:
        """Measure a single search against `max_retrieval_latency_ms`.

        Args:
            collection: Qdrant collection name.
            query_vector: Query embedding.
            top_k: Search cutoff.
            query: Optional original query text, stored in details.

        Returns:
            ValidationResult. passed if search time <= SLA. score is 1.0 when
            under SLA, otherwise sla / elapsed (clamped).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        sla = self.config.max_retrieval_latency_ms
        try:
            search_start = self._start_timer()
            hits = await self._integration.search(
                collection, query_vector, top_k=top_k
            )
            retrieval_ms = self._elapsed_ms(search_start)
            passed = retrieval_ms <= sla
            score = 1.0 if passed else _clamp(sla / retrieval_ms if retrieval_ms else 0.0)
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="retrieval_latency",
                threshold=0.0,
                details={
                    "query": query,
                    "collection": collection,
                    "k": top_k,
                    "retrieval_latency_ms": retrieval_ms,
                    "sla_ms": sla,
                    "hit_count": len(hits),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "retrieval_latency", exc, threshold=0.0
            )

        await self._write_audit(result)
        return result
