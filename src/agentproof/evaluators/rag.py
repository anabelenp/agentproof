"""RAGEvaluator — retrieval and generation quality for RAG pipelines.

Retrieval is scored with DeepEval ContextualRecallMetric and
ContextualPrecisionMetric. Generation reuses LLMEvaluator (faithfulness +
answer relevancy). An optional QdrantEvaluator supplies id-level retrieval
quality and live search for `evaluate_full_pipeline`.

Usage:
    evaluator = RAGEvaluator(config, audit_logger)
    result = await evaluator.evaluate_contextual_recall(
        query="What is our refund policy?",
        retrieved_docs=["Refunds are available within 30 days."],
        expected_output="Customers may request a refund within 30 days.",
    )
"""

from datetime import datetime, timezone
from typing import Any, Callable

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, RAGEvaluatorError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async
from agentproof.evaluators.llm import LLMEvaluator
from agentproof.integrations.qdrant import QdrantEvaluator

SUPPORTED_METRICS = (
    "contextual_recall",
    "recall",
    "contextual_precision",
    "precision",
    "faithfulness",
    "generation_faithfulness",
    "relevance",
    "generation_relevance",
    "generation",
)


class RAGEvaluator(BaseEvaluator):
    """Scores RAG retrieval (DeepEval) and generation (LLMEvaluator).

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with thresholds, judge model, and retry settings.
        audit_logger: AuditLogger that receives every result.
        llm: Optional LLMEvaluator used for generation metrics. Created from
            config and audit_logger if omitted.
        qdrant: Optional QdrantEvaluator used by evaluate_full_pipeline to
            search a collection and score id-level retrieval quality.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        llm: LLMEvaluator | None = None,
        qdrant: QdrantEvaluator | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._llm = llm
        self._qdrant = qdrant
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "RAGEvaluator"

    @property
    def llm(self) -> LLMEvaluator:
        """LLMEvaluator used for generation faithfulness and relevance."""
        if self._llm is None:
            self._llm = LLMEvaluator(self.config, self._audit)
        return self._llm

    async def evaluate(
        self,
        *,
        query: str | None = None,
        input: str | None = None,
        retrieved_docs: list[str] | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        expected_output: str | None = None,
        ground_truth_docs: list[str] | None = None,
        actual_output: str | None = None,
        output: str | None = None,
        metric: str = "contextual_recall",
        executive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a retrieval or generation metric.

        Args:
            query: User query. Alias of `input`.
            input: User query (TestRunner-friendly name).
            retrieved_docs: Documents the retriever returned.
            context: Alias of retrieved_docs.
            retrieval_context: Preferred alias of retrieved_docs.
            expected_output: Gold answer. Required for contextual metrics.
            ground_truth_docs: Joined into expected_output when it is omitted.
            actual_output: Generator response. Required for generation metrics.
            output: Alias of actual_output.
            metric: One of contextual_recall (default), contextual_precision,
                faithfulness, relevance, generation.
            executive: Forwarded to generation faithfulness.

        Returns:
            ValidationResult for the requested metric.
        """
        resolved_query = _first_text(query, input)
        docs = _first_list(retrieval_context, retrieved_docs, context)
        expected = expected_output
        if not expected and ground_truth_docs:
            expected = "\n".join(ground_truth_docs)
        response = _first_text(actual_output, output)

        key = metric.lower().strip()
        if key in {"contextual_recall", "recall"}:
            return await self.evaluate_contextual_recall(
                query=resolved_query,
                retrieved_docs=docs,
                expected_output=expected or "",
                actual_output=response or None,
            )
        if key in {"contextual_precision", "precision"}:
            return await self.evaluate_contextual_precision(
                query=resolved_query,
                retrieved_docs=docs,
                expected_output=expected or "",
                actual_output=response or None,
            )
        if key in {"faithfulness", "generation_faithfulness", "generation"}:
            return await self.evaluate_generation_faithfulness(
                query=resolved_query,
                actual_output=response or "",
                retrieved_docs=docs,
                executive=executive,
            )
        if key in {"relevance", "generation_relevance"}:
            return await self.evaluate_generation_relevance(
                query=resolved_query,
                actual_output=response or "",
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
        await self._write_audit(result, model=self.config.judge_model)
        return result

    async def evaluate_contextual_recall(
        self,
        *,
        query: str,
        retrieved_docs: list[str] | None,
        expected_output: str,
        actual_output: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether retrieved docs cover the gold answer.

        Empty retrieval is a quality failure (score 0, error unset).

        Args:
            query: User query.
            retrieved_docs: Documents the retriever returned.
            expected_output: Gold / ideal answer.
            actual_output: Optional generator response (not required by DeepEval).

        Returns:
            ValidationResult. passed when score >= contextual_recall_threshold
            (default 0.7). Higher is better.
        """
        docs = list(retrieved_docs or [])
        if not _nonempty_docs(docs):
            return await self._empty_retrieval_result(
                "contextual_recall", self.config.contextual_recall_threshold
            )

        def build_case() -> LLMTestCase:
            self._require_text(query, "query", "contextual_recall")
            self._require_text(expected_output, "expected_output", "contextual_recall")
            kwargs_case: dict[str, Any] = {
                "input": query,
                "expected_output": expected_output,
                "retrieval_context": docs,
            }
            if actual_output:
                kwargs_case["actual_output"] = actual_output
            return LLMTestCase(**kwargs_case)

        return await self._measure(
            metric_name="contextual_recall",
            threshold=self.config.contextual_recall_threshold,
            factory=ContextualRecallMetric,
            test_case_factory=build_case,
        )

    async def evaluate_contextual_precision(
        self,
        *,
        query: str,
        retrieved_docs: list[str] | None,
        expected_output: str,
        actual_output: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether relevant retrieved docs are ranked above irrelevant ones.

        Empty retrieval is a quality failure (score 0, error unset).

        Args:
            query: User query.
            retrieved_docs: Ranked documents the retriever returned.
            expected_output: Gold / ideal answer used to judge relevance.
            actual_output: Optional generator response.

        Returns:
            ValidationResult. passed when score >= contextual_precision_threshold
            (default 0.8). Higher is better.
        """
        docs = list(retrieved_docs or [])
        if not _nonempty_docs(docs):
            return await self._empty_retrieval_result(
                "contextual_precision", self.config.contextual_precision_threshold
            )

        def build_case() -> LLMTestCase:
            self._require_text(query, "query", "contextual_precision")
            self._require_text(expected_output, "expected_output", "contextual_precision")
            kwargs_case: dict[str, Any] = {
                "input": query,
                "expected_output": expected_output,
                "retrieval_context": docs,
            }
            if actual_output:
                kwargs_case["actual_output"] = actual_output
            return LLMTestCase(**kwargs_case)

        return await self._measure(
            metric_name="contextual_precision",
            threshold=self.config.contextual_precision_threshold,
            factory=ContextualPrecisionMetric,
            test_case_factory=build_case,
        )

    async def evaluate_retrieval(
        self,
        *,
        query: str,
        retrieved_docs: list[str],
        expected_output: str | None = None,
        ground_truth_docs: list[str] | None = None,
        actual_output: str | None = None,
        metric: str = "contextual_recall",
        **kwargs: Any,
    ) -> ValidationResult:
        """Score retrieval with DeepEval (default: contextual recall).

        Args:
            query: User query.
            retrieved_docs: Documents the retriever returned.
            expected_output: Gold answer. Required unless ground_truth_docs given.
            ground_truth_docs: Joined into expected_output when it is omitted.
            actual_output: Optional generator response.
            metric: "contextual_recall" (default) or "contextual_precision".

        Returns:
            ValidationResult for the requested retrieval metric.
        """
        expected = expected_output
        if not expected and ground_truth_docs:
            expected = "\n".join(ground_truth_docs)
        return await self.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            expected_output=expected,
            actual_output=actual_output,
            metric=metric,
        )

    async def evaluate_generation_faithfulness(
        self,
        *,
        query: str,
        actual_output: str,
        retrieved_docs: list[str] | None,
        executive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether the generated answer is grounded in retrieved docs.

        Delegates to LLMEvaluator.evaluate_faithfulness.

        Args:
            query: User query.
            actual_output: Generator response.
            retrieved_docs: Documents the generator used.
            executive: If True, use the executive faithfulness threshold.

        Returns:
            ValidationResult from LLMEvaluator (evaluator_name="LLMEvaluator").
        """
        return await self.llm.evaluate_faithfulness(
            input=query,
            output=actual_output,
            retrieval_context=list(retrieved_docs or []),
            executive=executive,
        )

    async def evaluate_generation_relevance(
        self,
        *,
        query: str,
        actual_output: str,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score whether the generated answer addresses the query.

        Delegates to LLMEvaluator.evaluate_relevance.

        Args:
            query: User query.
            actual_output: Generator response.

        Returns:
            ValidationResult from LLMEvaluator (evaluator_name="LLMEvaluator").
        """
        return await self.llm.evaluate_relevance(input=query, output=actual_output)

    async def evaluate_generation(
        self,
        *,
        query: str,
        actual_output: str,
        retrieved_docs: list[str] | None,
        executive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score generation faithfulness (primary RAG generation contract).

        Args:
            query: User query.
            actual_output: Generator response.
            retrieved_docs: Documents the generator used.
            executive: Forwarded to faithfulness.

        Returns:
            ValidationResult from evaluate_generation_faithfulness.
        """
        return await self.evaluate_generation_faithfulness(
            query=query,
            actual_output=actual_output,
            retrieved_docs=retrieved_docs,
            executive=executive,
        )

    async def evaluate_all(
        self,
        *,
        query: str,
        retrieved_docs: list[str] | None,
        expected_output: str,
        actual_output: str,
        executive: bool = False,
    ) -> list[ValidationResult]:
        """Run recall, precision, generation faithfulness, and relevance.

        Individual metric failures are captured as error/fail results and do
        not abort the rest.

        Args:
            query: User query.
            retrieved_docs: Documents the retriever returned.
            expected_output: Gold answer for contextual metrics.
            actual_output: Generator response.
            executive: Forwarded to generation faithfulness.

        Returns:
            Four ValidationResult objects, in the order above.
        """
        return [
            await self.evaluate_contextual_recall(
                query=query,
                retrieved_docs=retrieved_docs,
                expected_output=expected_output,
                actual_output=actual_output,
            ),
            await self.evaluate_contextual_precision(
                query=query,
                retrieved_docs=retrieved_docs,
                expected_output=expected_output,
                actual_output=actual_output,
            ),
            await self.evaluate_generation_faithfulness(
                query=query,
                actual_output=actual_output,
                retrieved_docs=retrieved_docs,
                executive=executive,
            ),
            await self.evaluate_generation_relevance(
                query=query,
                actual_output=actual_output,
            ),
        ]

    async def evaluate_full_pipeline(
        self,
        *,
        query: str,
        expected_output: str,
        actual_output: str,
        retrieved_docs: list[str] | None = None,
        collection: str | None = None,
        query_vector: list[float] | None = None,
        ground_truth_ids: list[str] | None = None,
        top_k: int = 5,
        executive: bool = False,
    ) -> list[ValidationResult]:
        """Run id-level retrieval (optional) plus RAG retrieval and generation.

        When `retrieved_docs` is omitted, searches Qdrant via the attached
        QdrantEvaluator (requires collection + query_vector).

        Args:
            query: User query.
            expected_output: Gold answer.
            actual_output: Generator response.
            retrieved_docs: Pre-fetched chunks. Searched from Qdrant if omitted.
            collection: Qdrant collection for live search / id-level quality.
            query_vector: Embedding used to search Qdrant.
            ground_truth_ids: Relevant ids for QdrantEvaluator retrieval_quality.
            top_k: Qdrant search cutoff.
            executive: Forwarded to generation faithfulness.

        Returns:
            ValidationResult list. Qdrant retrieval_quality is first when
            ground_truth_ids and a QdrantEvaluator are present, followed by
            the four RAG metrics from evaluate_all.
        """
        results: list[ValidationResult] = []
        docs = list(retrieved_docs) if retrieved_docs is not None else None

        if docs is None:
            if self._qdrant is None or not collection or not query_vector:
                audit_id = self._new_audit_id()
                start = self._start_timer()
                exc = RAGEvaluatorError(
                    "retrieved_docs is required when Qdrant search inputs are missing",
                    evaluator_name=self.name,
                    metric="full_pipeline",
                )
                result = self._error_result(
                    audit_id, start, "full_pipeline", exc, threshold=0.0
                )
                await self._write_audit(result, model=self.config.judge_model)
                return [result]
            try:
                hits = await self._qdrant.integration.search(
                    collection, query_vector, top_k=top_k
                )
                docs = [hit.text for hit in hits if hit.text]
            except AuditError:
                raise
            except Exception as exc:
                audit_id = self._new_audit_id()
                start = self._start_timer()
                result = self._error_result(
                    audit_id, start, "full_pipeline", exc, threshold=0.0
                )
                await self._write_audit(result, model=self.config.judge_model)
                return [result]

        if self._qdrant is not None and collection and query_vector and ground_truth_ids:
            results.append(
                await self._qdrant.evaluate_retrieval_quality(
                    collection=collection,
                    query_vector=query_vector,
                    ground_truth_ids=ground_truth_ids,
                    top_k=top_k,
                    query=query,
                )
            )

        results.extend(
            await self.evaluate_all(
                query=query,
                retrieved_docs=docs,
                expected_output=expected_output,
                actual_output=actual_output,
                executive=executive,
            )
        )
        return results

    async def _empty_retrieval_result(
        self, metric_name: str, threshold: float
    ) -> ValidationResult:
        """Quality failure for an empty retriever response (not an evaluator error).

        Args:
            metric_name: contextual_recall or contextual_precision.
            threshold: Pass/fail threshold that was in effect.

        Returns:
            ValidationResult with passed=False, score=0.0, error unset.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=metric_name,
            threshold=threshold,
            details={"reason": "empty retrieval", "empty_retrieval": True},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
        )
        await self._write_audit(result, model=self.config.judge_model)
        return result

    async def _measure(
        self,
        *,
        metric_name: str,
        threshold: float,
        factory: Callable[..., Any],
        test_case_factory: Callable[[], LLMTestCase],
    ) -> ValidationResult:
        """Construct a DeepEval metric, measure asynchronously, and audit.

        Args:
            metric_name: Stored on ValidationResult.metric.
            threshold: AgentProof pass/fail threshold.
            factory: DeepEval metric class.
            test_case_factory: Builds the LLMTestCase; may raise RAGEvaluatorError.

        Returns:
            ValidationResult. Evaluation exceptions become error results.
            AuditError propagates.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            test_case = test_case_factory()
            metric = factory(
                threshold=threshold,
                model=self._judge_model(),
                include_reason=True,
            )

            async def _run() -> Any:
                return await metric.a_measure(test_case)

            await retry_async(self._retry)(_run)()

            raw = metric.score
            if raw is None:
                raise RAGEvaluatorError(
                    "DeepEval returned no score",
                    evaluator_name=self.name,
                    metric=metric_name,
                )
            score = _clamp(float(raw))
            passed = score >= threshold
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric=metric_name,
                threshold=threshold,
                details={
                    "reason": getattr(metric, "reason", None) or "",
                    "deepeval_score": score,
                    "deepeval_success": bool(getattr(metric, "success", passed)),
                    "score_interpretation": "higher is better",
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except RetryExhaustedError as exc:
            cause = exc.last_exception or exc
            wrapped = RAGEvaluatorError(
                f"failed after {self._retry.max_attempts} attempts: "
                f"{type(cause).__name__}: {cause}",
                evaluator_name=self.name,
                metric=metric_name,
            )
            result = self._error_result(
                audit_id, start, metric_name, wrapped, threshold=threshold
            )
        except Exception as exc:
            result = self._error_result(
                audit_id, start, metric_name, exc, threshold=threshold
            )

        await self._write_audit(result, model=self.config.judge_model)
        return result

    def _judge_model(self) -> str | AnthropicModel:
        """Return the DeepEval judge: AnthropicModel for Claude, else a name string.

        Returns:
            An AnthropicModel when the configured judge is a Claude model
            and `anthropic_api_key` is set; otherwise the model name string.
        """
        name = self.config.judge_model
        if name.startswith("claude") and self.config.anthropic_api_key:
            return AnthropicModel(
                model=name,
                api_key=self.config.anthropic_api_key,
                temperature=0,
            )
        return name

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise RAGEvaluatorError if `value` is empty or whitespace.

        Args:
            value: Field value to validate.
            field: Field name for the error message.
            metric: Metric being evaluated, stored on the error.

        Raises:
            RAGEvaluatorError: If value is empty or whitespace-only.
        """
        if not isinstance(value, str) or not value.strip():
            raise RAGEvaluatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _first_text(*values: str | None) -> str:
    """Return the first non-None string (may be empty)."""
    for value in values:
        if value is not None:
            return value
    return ""


def _first_list(*values: list[str] | None) -> list[str]:
    """Return the first non-None list (may be empty)."""
    for value in values:
        if value is not None:
            return list(value)
    return []


def _nonempty_docs(docs: list[str]) -> bool:
    """True if at least one document has non-whitespace text."""
    return any(isinstance(doc, str) and doc.strip() for doc in docs)
