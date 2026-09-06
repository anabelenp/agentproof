"""StreamingValidator — TTFT, throughput, completeness, and failure handling.

Scores async streaming responses (SSE / chunked HTTP) against configured SLAs:

- TTFT < `config.max_ttft_seconds` (default 2s)
- Throughput > `config.min_token_throughput` tokens/second (default 20)
- Completeness — stream ends with a done marker and non-empty content
- Graceful degradation — model unavailability surfaces as an error, not silence
- Error handling — mid-stream interruptions produce an error state

Unit tests pass pre-recorded `chunks` so no network is required. Live checks
POST `payload` to `endpoint` via httpx and parse SSE `data:` lines.

Usage:
    validator = StreamingValidator(config, audit_logger)
    result = await validator.validate_ttft(
        chunks=["Hello", " world"],
        first_token_at=0.12,
        ended_at=0.80,
    )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, StreamingValidatorError

SUPPORTED_METRICS = (
    "ttft",
    "throughput",
    "completeness",
    "graceful_degradation",
    "error_handling",
)

DONE_MARKERS = ("[DONE]", "data: [DONE]")


@dataclass
class StreamSample:
    """A recorded or live streaming response.

    Args:
        text: Concatenated assistant text (SSE payloads joined).
        t0: Request start time (perf_counter).
        first_token_at: Time of first non-empty token, or None.
        ended_at: Time the stream closed.
        done: True if a terminal done marker was observed.
        status_code: HTTP status if collected over the network.
        error: Transport or application error; None on a clean stream.
        token_count: Whitespace-delimited token count of `text`.
        chunk_count: Number of chunks received.
    """

    text: str
    t0: float
    first_token_at: float | None
    ended_at: float
    done: bool = False
    status_code: int | None = None
    error: str | None = None
    token_count: int = 0
    chunk_count: int = 0

    @property
    def ttft_seconds(self) -> float | None:
        """Seconds from request start to first token, or None if none arrived."""
        if self.first_token_at is None:
            return None
        return max(0.0, self.first_token_at - self.t0)

    @property
    def duration_seconds(self) -> float:
        """Seconds from request start to stream close."""
        return max(0.0, self.ended_at - self.t0)

    @property
    def tokens_per_second(self) -> float:
        """End-to-end token throughput (tokens / duration)."""
        duration = self.duration_seconds
        if duration <= 0.0:
            return 0.0
        return self.token_count / duration


class StreamingValidator(BaseEvaluator):
    """Validates streaming LLM responses against TTFT and throughput SLAs.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with max_ttft_seconds and min_token_throughput.
        audit_logger: AuditLogger that receives every result.
        client: Optional httpx.AsyncClient. Created per-call if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._client = client

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "StreamingValidator"

    async def evaluate(
        self,
        *,
        metric: str = "ttft",
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        mock_failure: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a streaming metric.

        Args:
            metric: One of ttft (default), throughput, completeness,
                graceful_degradation, error_handling.
            endpoint: URL to POST for a live stream.
            payload: JSON body for the live request.
            chunks: Pre-recorded text chunks (unit tests).
            headers: Optional HTTP headers.
            t0 / first_token_at / ended_at: Timing for pre-recorded chunks,
                in seconds relative to the same clock.
            done: Whether a done marker was observed (pre-recorded).
            status_code: HTTP status for pre-recorded / mocked streams.
            error: Pre-recorded error string.
            mock_failure: For graceful_degradation, treat the stream as a
                provider failure even if the HTTP call is skipped.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        sample_kwargs: dict[str, Any] = dict(
            endpoint=endpoint,
            payload=payload,
            chunks=chunks,
            headers=headers,
            t0=t0,
            first_token_at=first_token_at,
            ended_at=ended_at,
            done=done,
            status_code=status_code,
            error=error,
        )
        if key == "ttft":
            return await self.validate_ttft(**sample_kwargs)
        if key == "throughput":
            return await self.validate_throughput(**sample_kwargs)
        if key == "completeness":
            return await self.validate_completeness(**sample_kwargs)
        if key == "graceful_degradation":
            return await self.validate_graceful_degradation(
                mock_failure=mock_failure, **sample_kwargs
            )
        if key == "error_handling":
            return await self.validate_error_handling(**sample_kwargs)

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

    async def validate_ttft(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score time-to-first-token against `max_ttft_seconds`.

        Args:
            endpoint / payload / chunks / headers: Stream source.
            t0 / first_token_at / ended_at / done / status_code / error:
                Timing and status for pre-recorded streams.

        Returns:
            ValidationResult. passed if TTFT <= SLA. score is 1.0 under SLA,
            otherwise sla / ttft (clamped).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        sla = self.config.max_ttft_seconds
        try:
            sample = await self._collect_sample(
                endpoint=endpoint,
                payload=payload,
                chunks=chunks,
                headers=headers,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
            if sample.error and sample.ttft_seconds is None:
                raise StreamingValidatorError(
                    f"stream failed before first token: {sample.error}",
                    evaluator_name=self.name,
                    metric="ttft",
                )
            ttft = sample.ttft_seconds
            if ttft is None:
                passed = False
                score = 0.0
            else:
                passed = ttft <= sla
                score = 1.0 if passed else _clamp(sla / ttft if ttft else 0.0)
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="ttft",
                threshold=0.0,
                details={
                    "ttft_seconds": ttft,
                    "sla_seconds": sla,
                    "token_count": sample.token_count,
                    "chunk_count": sample.chunk_count,
                    "status_code": sample.status_code,
                    "error": sample.error,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(audit_id, start, "ttft", exc, threshold=0.0)

        await self._write_audit(result)
        return result

    async def validate_throughput(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Score tokens/second against `min_token_throughput`.

        Args:
            endpoint / payload / chunks / headers: Stream source.
            t0 / first_token_at / ended_at / done / status_code / error:
                Timing and status for pre-recorded streams.

        Returns:
            ValidationResult. passed if tokens/sec >= SLA. score is
            min(1.0, measured / sla).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        sla = self.config.min_token_throughput
        try:
            sample = await self._collect_sample(
                endpoint=endpoint,
                payload=payload,
                chunks=chunks,
                headers=headers,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
            if sample.error:
                raise StreamingValidatorError(
                    f"stream failed: {sample.error}",
                    evaluator_name=self.name,
                    metric="throughput",
                )
            tps = sample.tokens_per_second
            passed = tps >= sla
            score = _clamp(tps / sla) if sla else 0.0
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="throughput",
                threshold=0.0,
                details={
                    "tokens_per_second": tps,
                    "sla_tokens_per_second": sla,
                    "token_count": sample.token_count,
                    "duration_seconds": sample.duration_seconds,
                    "status_code": sample.status_code,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "throughput", exc, threshold=0.0
            )

        await self._write_audit(result)
        return result

    async def validate_completeness(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Detect truncated streams (no done marker, empty body, or error).

        Args:
            endpoint / payload / chunks / headers: Stream source.
            t0 / first_token_at / ended_at / done / status_code / error:
                Timing and status for pre-recorded streams.

        Returns:
            ValidationResult. passed if content is non-empty, a done marker
            was seen, and no stream error was recorded.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            sample = await self._collect_sample(
                endpoint=endpoint,
                payload=payload,
                chunks=chunks,
                headers=headers,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
            has_text = bool(sample.text.strip())
            passed = has_text and sample.done and sample.error is None
            score = 1.0 if passed else 0.0
            result = ValidationResult(
                passed=passed,
                score=score,
                evaluator_name=self.name,
                metric="completeness",
                threshold=1.0,
                details={
                    "has_text": has_text,
                    "done": sample.done,
                    "error": sample.error,
                    "token_count": sample.token_count,
                    "chunk_count": sample.chunk_count,
                    "status_code": sample.status_code,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "completeness", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_graceful_degradation(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        mock_failure: bool = True,
        **kwargs: Any,
    ) -> ValidationResult:
        """Pass when a provider failure is visible, not a silent empty 200.

        Args:
            mock_failure: When True and no live error is supplied, treat the
                sample as a failed provider (unit-test helper).
            Other args: Stream source / pre-recorded timing.

        Returns:
            ValidationResult. passed if an error or non-2xx status is present.
            Silent success (2xx, empty body, no error) fails.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if (
                mock_failure
                and error is None
                and status_code is None
                and chunks is None
                and not endpoint
            ):
                error = "model unavailable"
                status_code = 503
                chunks = []
            sample = await self._collect_sample(
                endpoint=endpoint,
                payload=payload,
                chunks=chunks,
                headers=headers,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
            visible_failure = bool(sample.error) or (
                sample.status_code is not None and sample.status_code >= 400
            )
            silent = (
                (sample.status_code is None or 200 <= sample.status_code < 300)
                and not sample.error
                and not sample.text.strip()
            )
            passed = visible_failure and not silent
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="graceful_degradation",
                threshold=1.0,
                details={
                    "visible_failure": visible_failure,
                    "silent_failure": silent,
                    "status_code": sample.status_code,
                    "error": sample.error,
                    "token_count": sample.token_count,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "graceful_degradation", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_error_handling(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Pass when a mid-stream interruption is recorded as an error.

        A stream that returns 200 with partial text and no error (silent
        truncation) fails.

        Args:
            Stream source / pre-recorded timing, plus `error` for an injected
            interruption.

        Returns:
            ValidationResult. passed if `sample.error` is set.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            sample = await self._collect_sample(
                endpoint=endpoint,
                payload=payload,
                chunks=chunks,
                headers=headers,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
            surfaced = bool(sample.error)
            silent_partial = (
                bool(sample.text.strip())
                and not sample.done
                and not sample.error
                and (sample.status_code is None or 200 <= sample.status_code < 300)
            )
            passed = surfaced and not silent_partial
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="error_handling",
                threshold=1.0,
                details={
                    "error_surfaced": surfaced,
                    "silent_partial": silent_partial,
                    "error": sample.error,
                    "done": sample.done,
                    "token_count": sample.token_count,
                    "status_code": sample.status_code,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "error_handling", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        endpoint: str | None = None,
        payload: dict[str, Any] | None = None,
        chunks: list[str] | None = None,
        headers: dict[str, str] | None = None,
        t0: float = 0.0,
        first_token_at: float | None = None,
        ended_at: float | None = None,
        done: bool | None = None,
        status_code: int | None = None,
        error: str | None = None,
    ) -> list[ValidationResult]:
        """Collect the stream once, then score TTFT, throughput, and completeness.

        Args:
            Same stream source as the individual validators.

        Returns:
            Three ValidationResult objects. Individual failures do not abort
            the rest.
        """
        sample = await self._collect_sample(
            endpoint=endpoint,
            payload=payload,
            chunks=chunks,
            headers=headers,
            t0=t0,
            first_token_at=first_token_at,
            ended_at=ended_at,
            done=done,
            status_code=status_code,
            error=error,
        )
        # Replay the collected sample as pre-recorded chunks so we do not
        # hit the network three times.
        replay = dict(
            chunks=[sample.text] if sample.text else [],
            t0=sample.t0,
            first_token_at=sample.first_token_at,
            ended_at=sample.ended_at,
            done=sample.done,
            status_code=sample.status_code,
            error=sample.error,
        )
        return [
            await self.validate_ttft(**replay),
            await self.validate_throughput(**replay),
            await self.validate_completeness(**replay),
        ]

    async def _collect_sample(
        self,
        *,
        endpoint: str | None,
        payload: dict[str, Any] | None,
        chunks: list[str] | None,
        headers: dict[str, str] | None,
        t0: float,
        first_token_at: float | None,
        ended_at: float | None,
        done: bool | None,
        status_code: int | None,
        error: str | None,
    ) -> StreamSample:
        """Build a StreamSample from pre-recorded chunks or a live HTTP POST.

        Args:
            endpoint: Live URL. Ignored when `chunks` is provided.
            payload: JSON body for the live request.
            chunks: Pre-recorded text pieces.
            headers: Optional HTTP headers.
            t0 / first_token_at / ended_at: Timing for pre-recorded streams.
            done / status_code / error: Status for pre-recorded streams.

        Returns:
            StreamSample. Network failures are captured in `error`, not raised.

        Raises:
            StreamingValidatorError: Neither chunks nor endpoint was provided.
        """
        if chunks is not None:
            return _sample_from_chunks(
                chunks,
                t0=t0,
                first_token_at=first_token_at,
                ended_at=ended_at,
                done=done,
                status_code=status_code,
                error=error,
            )
        if not endpoint:
            raise StreamingValidatorError(
                "chunks or endpoint is required",
                evaluator_name=self.name,
                metric="stream",
            )
        return await self._collect_http(
            endpoint, payload or {}, headers=headers or {}
        )

    async def _collect_http(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
    ) -> StreamSample:
        """POST to `endpoint` and collect an SSE or chunked text stream.

        Args:
            endpoint: URL to stream from.
            payload: JSON body.
            headers: Extra request headers.

        Returns:
            StreamSample with timings measured via perf_counter.
        """
        import time

        t0 = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        done = False
        status_code: int | None = None
        error: str | None = None

        client = self._client
        owns_client = client is None
        if client is None:
            timeout = max(30.0, self.config.max_ttft_seconds * 10.0)
            client = httpx.AsyncClient(timeout=timeout)

        try:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers or None,
            ) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    error = f"HTTP {response.status_code}"
                async for raw in response.aiter_text():
                    text, chunk_done = parse_sse_fragment(raw)
                    if chunk_done:
                        done = True
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(text)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if owns_client:
                await client.aclose()

        ended_at = time.perf_counter()
        body = "".join(pieces)
        if not done:
            done = body_has_done_marker(body)
        return StreamSample(
            text=strip_done_markers(body),
            t0=t0,
            first_token_at=first_token_at,
            ended_at=ended_at,
            done=done,
            status_code=status_code,
            error=error,
            token_count=count_tokens(strip_done_markers(body)),
            chunk_count=len(pieces),
        )


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens.

    Args:
        text: Stream text.

    Returns:
        Token count, or 0 for empty/whitespace.
    """
    return len(text.split()) if text.strip() else 0


def parse_sse_fragment(raw: str) -> tuple[str, bool]:
    """Parse one SSE (or raw text) fragment.

    Args:
        raw: Chunk from `aiter_text()`.

    Returns:
        (payload_text, done) where done is True if a [DONE] marker was seen.
    """
    if not raw:
        return "", False
    done = body_has_done_marker(raw)
    if "data:" in raw:
        parts: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            payload = stripped[5:].strip()
            if payload in {"[DONE]", "DONE"}:
                done = True
                continue
            parts.append(payload)
        return "".join(parts), done
    return strip_done_markers(raw), done


def body_has_done_marker(text: str) -> bool:
    """True if `text` contains an SSE/OpenAI done sentinel."""
    return any(marker in text for marker in DONE_MARKERS)


def strip_done_markers(text: str) -> str:
    """Remove done sentinels from concatenated stream text."""
    cleaned = text
    for marker in DONE_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned


def _sample_from_chunks(
    chunks: list[str],
    *,
    t0: float,
    first_token_at: float | None,
    ended_at: float | None,
    done: bool | None,
    status_code: int | None,
    error: str | None,
) -> StreamSample:
    """Build a StreamSample from pre-recorded chunks.

    Args:
        chunks: Ordered text pieces. Done markers inside chunks set `done`.
        t0: Request start.
        first_token_at: First-token time; defaults to t0 if any text exists.
        ended_at: Stream end; defaults to first_token_at or t0.
        done: Explicit done flag; inferred from chunks when omitted.
        status_code: Optional HTTP status.
        error: Optional error string.

    Returns:
        StreamSample.
    """
    combined_raw = "".join(chunks)
    inferred_done = body_has_done_marker(combined_raw) if done is None else done
    text = strip_done_markers(combined_raw)
    if first_token_at is None and text.strip():
        first_token_at = t0
    if ended_at is None:
        ended_at = first_token_at if first_token_at is not None else t0
    return StreamSample(
        text=text,
        t0=t0,
        first_token_at=first_token_at,
        ended_at=ended_at,
        done=bool(inferred_done),
        status_code=status_code,
        error=error,
        token_count=count_tokens(text),
        chunk_count=len(chunks),
    )


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
