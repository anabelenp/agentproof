"""IngestionValidator — Nango connectors and file ingestion.

Scores connector reliability, silent-failure detection, completeness,
schema drift, OAuth refresh, rate-limit backoff, audit continuity, and
file ingestion. Unit tests pass in-memory records so no Nango account is
required.

Usage:
    validator = IngestionValidator(config, audit_logger, integration=nango)
    result = await validator.validate_connector_reliability(
        connector_id="salesforce-1",
        expected_record_count=3,
        sample_fields=["id", "email"],
        records=ingested,
    )
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IngestionValidatorError
from agentproof.integrations.nango import NangoIntegration

SUPPORTED_METRICS = (
    "connector_reliability",
    "silent_failure",
    "completeness",
    "schema_drift",
    "auth_refresh",
    "rate_limit",
    "audit_continuity",
    "file_ingestion",
)

INJECTED_FAILURES = ("drop_records", "malform_payload", "auth_expire")
FILE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".txt", ".md"}


def record_id(row: dict[str, Any]) -> str:
    """Best-effort record id from common connector field names.

    Args:
        row: Ingested record.

    Returns:
        String id, or "".
    """
    for key in ("id", "record_id", "source_id", "external_id", "Id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def field_present(row: dict[str, Any], field: str) -> bool:
    """True when `field` exists on `row` and is not None."""
    if field not in row:
        return False
    return row[field] is not None


def type_name(value: Any) -> str:
    """Stable type label for schema-drift comparison."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def classify_drift(
    original_schema: dict[str, Any],
    drifted_schema: dict[str, Any],
) -> dict[str, Any]:
    """Diff two field→example-value (or field→type) maps.

    Args:
        original_schema: Source field map before the change.
        drifted_schema: Source field map after the change.

    Returns:
        added, removed, and type_changes lists.
    """
    orig = set(original_schema)
    new = set(drifted_schema)
    type_changes = [
        key
        for key in orig & new
        if type_name(original_schema[key]) != type_name(drifted_schema[key])
    ]
    return {
        "added": sorted(new - orig),
        "removed": sorted(orig - new),
        "type_changes": sorted(type_changes),
    }


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class IngestionValidator(BaseEvaluator):
    """Scores Nango connector ingestion and file-ingestion pipelines.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with Nango URL and completeness threshold.
        audit_logger: AuditLogger that receives every result.
        integration: Optional NangoIntegration. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        integration: NangoIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._integration = integration or NangoIntegration(config)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "IngestionValidator"

    @property
    def integration(self) -> NangoIntegration:
        """The NangoIntegration used for live connector calls."""
        return self._integration

    async def evaluate(
        self,
        *,
        metric: str = "connector_reliability",
        connector_id: str = "",
        expected_record_count: int | None = None,
        sample_fields: list[str] | None = None,
        records: list[dict[str, Any]] | None = None,
        model: str = "",
        provider_config_key: str = "",
        injected_failure: str = "",
        error: str | None = None,
        status_code: int | None = None,
        expected_ids: list[str] | None = None,
        original_schema: dict[str, Any] | None = None,
        drifted_schema: dict[str, Any] | None = None,
        handling: str = "",
        required_fields: list[str] | None = None,
        token_refreshed: bool | None = None,
        data_loss: bool | None = None,
        status_codes: list[int] | None = None,
        retried: bool | None = None,
        trail: list[dict[str, Any]] | None = None,
        sync_run_id: str = "",
        filename: str = "",
        extracted: list[dict[str, Any]] | None = None,
        encoding_error: bool = False,
        crashed: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to an ingestion check.

        Args:
            metric: connector_reliability (default), silent_failure,
                completeness, schema_drift, auth_refresh, rate_limit,
                audit_continuity, file_ingestion.
            connector_id: Nango connection id.
            expected_record_count / sample_fields / records: Reliability.
            model / provider_config_key: Live Nango list_records args.
            injected_failure / error / status_code: Silent-failure scenario.
            expected_ids: Completeness ground truth.
            original_schema / drifted_schema / handling / required_fields:
                Schema-drift scenario.
            token_refreshed / data_loss: Auth refresh outcome.
            status_codes / retried: Rate-limit sequence.
            trail / sync_run_id: Audit continuity hops.
            filename / extracted / encoding_error / crashed: File ingestion.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        if key == "connector_reliability":
            return await self.validate_connector_reliability(
                connector_id=connector_id,
                expected_record_count=expected_record_count if expected_record_count is not None else 0,
                sample_fields=sample_fields or [],
                records=records,
                model=model,
                provider_config_key=provider_config_key,
            )
        if key == "silent_failure":
            return await self.validate_silent_failure(
                connector_id=connector_id,
                injected_failure=injected_failure,
                records=records,
                error=error,
                status_code=status_code,
            )
        if key == "completeness":
            return await self.validate_completeness(
                connector_id=connector_id,
                expected_ids=expected_ids or [],
                sample_fields=sample_fields or [],
                records=records,
                model=model,
            )
        if key == "schema_drift":
            return await self.validate_schema_drift(
                connector_id=connector_id,
                original_schema=original_schema or {},
                drifted_schema=drifted_schema or {},
                handling=handling,
                required_fields=required_fields or [],
                crashed=crashed,
            )
        if key == "auth_refresh":
            return await self.validate_auth_refresh(
                connector_id=connector_id,
                token_refreshed=token_refreshed,
                data_loss=data_loss if data_loss is not None else False,
                records=records,
            )
        if key == "rate_limit":
            return await self.validate_rate_limit(
                connector_id=connector_id,
                status_codes=status_codes or [],
                retried=retried if retried is not None else False,
            )
        if key == "audit_continuity":
            return await self.validate_audit_continuity(
                connector_id=connector_id,
                sync_run_id=sync_run_id,
                trail=trail or [],
            )
        if key == "file_ingestion":
            return await self.validate_file_ingestion(
                filename=filename,
                extracted=extracted,
                expected_record_count=expected_record_count,
                sample_fields=sample_fields or [],
                encoding_error=encoding_error,
                crashed=crashed,
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

    async def validate_connector_reliability(
        self,
        *,
        connector_id: str,
        expected_record_count: int,
        sample_fields: list[str],
        records: list[dict[str, Any]] | None = None,
        model: str = "",
        provider_config_key: str = "",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a connector sync produced the expected rows and fields.

        Args:
            connector_id: Nango connection id.
            expected_record_count: Expected ingested row count.
            sample_fields: Fields that must be present and non-null.
            records: In-memory rows. Loaded from Nango when omitted.
            model / provider_config_key: Live list_records args.

        Returns:
            ValidationResult. score is the fraction of checks that passed
            (count match + each sample field on each row).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "connector_reliability")
            rows = await self._load_records(
                records, connector_id, model, provider_config_key
            )
            count_ok = len(rows) == expected_record_count
            missing_fields: list[str] = []
            for field in sample_fields:
                if any(not field_present(row, field) for row in rows) or not rows:
                    missing_fields.append(field)
            field_hits = len(sample_fields) - len(missing_fields)
            checks = 1 + len(sample_fields)
            passed_checks = (1 if count_ok else 0) + field_hits
            score = _clamp(passed_checks / checks) if checks else 1.0
            result = ValidationResult(
                passed=count_ok and not missing_fields,
                score=score,
                evaluator_name=self.name,
                metric="connector_reliability",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "expected_count": expected_record_count,
                    "actual_count": len(rows),
                    "sample_fields": list(sample_fields),
                    "missing_fields": missing_fields,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "connector_reliability", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_silent_failure(
        self,
        *,
        connector_id: str,
        injected_failure: str,
        records: list[dict[str, Any]] | None = None,
        error: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Fail when an injected connector failure is swallowed.

        Args:
            connector_id: Nango connection id.
            injected_failure: drop_records, malform_payload, or auth_expire.
            records: Rows after the failure (may be short or malformed).
            error: Surfaced error string, if any.
            status_code: HTTP status if the connector returned one.

        Returns:
            ValidationResult. score is 1.0 when the failure is visible.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "silent_failure")
            kind = injected_failure.strip().lower()
            if kind not in INJECTED_FAILURES:
                raise IngestionValidatorError(
                    "injected_failure must be drop_records, malform_payload, or auth_expire",
                    evaluator_name=self.name,
                    metric="silent_failure",
                )
            rows = list(records or [])
            malformed = [
                record_id(row) or f"index-{index}"
                for index, row in enumerate(rows)
                if not isinstance(row, dict) or row.get("_malformed") or row.get("malformed")
            ]
            surfaced = bool(error) or (
                status_code is not None and int(status_code) >= 400
            )
            if kind == "drop_records":
                # A drop with no error and remaining rows looks successful.
                silent = not surfaced
            elif kind == "malform_payload":
                silent = bool(malformed) and not surfaced
            else:  # auth_expire
                silent = not surfaced
            passed = not silent
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="silent_failure",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "injected_failure": kind,
                    "surfaced": surfaced,
                    "silent_failure": silent,
                    "status_code": status_code,
                    "record_count": len(rows),
                    "malformed_ids": malformed,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "silent_failure", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_completeness(
        self,
        *,
        connector_id: str,
        expected_ids: list[str],
        sample_fields: list[str] | None = None,
        records: list[dict[str, Any]] | None = None,
        model: str = "",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm every expected source id was ingested and mapped.

        Args:
            connector_id: Nango connection id.
            expected_ids: Source-system ids that must appear.
            sample_fields: Fields that must be mapped on each found row.
            records: In-memory rows. Loaded from Nango when omitted.
            model: Live list_records model.

        Returns:
            ValidationResult. score is found/expected (field mapping can
            lower it). passed when score >= ingestion_completeness_threshold
            and no expected id is missing.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        threshold = self.config.ingestion_completeness_threshold
        try:
            self._require_text(connector_id, "connector_id", "completeness")
            if not expected_ids:
                raise IngestionValidatorError(
                    "completeness requires non-empty expected_ids",
                    evaluator_name=self.name,
                    metric="completeness",
                )
            rows = await self._load_records(records, connector_id, model, "")
            got = {record_id(row) for row in rows if record_id(row)}
            expected = [str(item) for item in expected_ids]
            missing = [item for item in expected if item not in got]
            fields = list(sample_fields or [])
            unmapped: list[str] = []
            if fields:
                by_id = {record_id(row): row for row in rows if record_id(row)}
                for ident in expected:
                    row = by_id.get(ident)
                    if row is None:
                        continue
                    for field in fields:
                        if not field_present(row, field):
                            unmapped.append(f"{ident}.{field}")
            found = len(expected) - len(missing)
            id_score = found / len(expected)
            if fields and expected:
                field_checks = len(expected) * len(fields)
                field_hits = field_checks - len(unmapped) - len(missing) * len(fields)
                field_score = field_hits / field_checks if field_checks else 1.0
                score = _clamp((id_score + max(field_score, 0.0)) / 2.0)
            else:
                score = _clamp(id_score)
            result = ValidationResult(
                passed=score >= threshold and not missing,
                score=score,
                evaluator_name=self.name,
                metric="completeness",
                threshold=threshold,
                details={
                    "connector_id": connector_id,
                    "expected_ids": expected,
                    "missing_ids": missing,
                    "unmapped_fields": unmapped,
                    "actual_count": len(rows),
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "completeness", exc, threshold=threshold
            )

        await self._write_audit(result)
        return result

    async def validate_schema_drift(
        self,
        *,
        connector_id: str,
        original_schema: dict[str, Any],
        drifted_schema: dict[str, Any],
        handling: str = "",
        required_fields: list[str] | None = None,
        crashed: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a source schema change is alerted, not a crash or silent drop.

        Args:
            connector_id: Nango connection id.
            original_schema: Field map before the change.
            drifted_schema: Field map after the change.
            handling: alert, crash, or ignore (how the pipeline reacted).
            required_fields: Fields that must not disappear silently.
            crashed: If True, treat handling as crash.

        Returns:
            ValidationResult. passed when the pipeline did not crash and
            required-field loss was alerted.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "schema_drift")
            if not original_schema or not drifted_schema:
                raise IngestionValidatorError(
                    "schema_drift requires original_schema and drifted_schema",
                    evaluator_name=self.name,
                    metric="schema_drift",
                )
            mode = "crash" if crashed else handling.strip().lower()
            drift = classify_drift(original_schema, drifted_schema)
            required = list(required_fields or [])
            lost_required = [field for field in required if field in drift["removed"]]
            typed_required = [field for field in required if field in drift["type_changes"]]
            silent = bool(lost_required or typed_required) and mode in {"", "ignore"}
            crashed_run = mode == "crash"
            passed = not crashed_run and not silent
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="schema_drift",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "handling": mode or "unspecified",
                    "added": drift["added"],
                    "removed": drift["removed"],
                    "type_changes": drift["type_changes"],
                    "lost_required": lost_required,
                    "typed_required": typed_required,
                    "silent": silent,
                    "crashed": crashed_run,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "schema_drift", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_auth_refresh(
        self,
        *,
        connector_id: str,
        token_refreshed: bool | None = None,
        data_loss: bool = False,
        records: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm OAuth refresh succeeded without dropping ingested data.

        Args:
            connector_id: Nango connection id.
            token_refreshed: Outcome flag. Loaded from connection JSON when
                omitted (`credentials.raw.refresh` or `refreshed`).
            data_loss: True if records disappeared across the refresh.
            records: Optional rows after refresh (empty implies data_loss).

        Returns:
            ValidationResult. passed when refreshed and no data loss.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "auth_refresh")
            refreshed = token_refreshed
            if refreshed is None:
                connection = await self._integration.get_connection(connector_id)
                refreshed = bool(
                    connection.get("refreshed")
                    or connection.get("token_refreshed")
                    or (connection.get("credentials") or {}).get("refreshed")
                )
            lost = data_loss or (records is not None and len(records) == 0)
            passed = bool(refreshed) and not lost
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="auth_refresh",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "token_refreshed": bool(refreshed),
                    "data_loss": lost,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "auth_refresh", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_rate_limit(
        self,
        *,
        connector_id: str,
        status_codes: list[int],
        retried: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm 429s are retried rather than treated as a hard failure.

        Args:
            connector_id: Nango connection id.
            status_codes: Ordered HTTP statuses observed (e.g. [429, 200]).
            retried: True if the client retried after 429.

        Returns:
            ValidationResult. passed when a 429 is followed by success or
            `retried` is True and the last status is < 400.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "rate_limit")
            if not status_codes:
                raise IngestionValidatorError(
                    "rate_limit requires a non-empty status_codes list",
                    evaluator_name=self.name,
                    metric="rate_limit",
                )
            saw_429 = 429 in status_codes
            last = int(status_codes[-1])
            recovered = last < 400
            backed_off = retried or (
                saw_429 and any(code != 429 for code in status_codes[1:])
            )
            passed = (not saw_429 and recovered) or (saw_429 and backed_off and recovered)
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="rate_limit",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "status_codes": [int(code) for code in status_codes],
                    "saw_429": saw_429,
                    "retried": backed_off,
                    "recovered": recovered,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "rate_limit", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_audit_continuity(
        self,
        *,
        connector_id: str,
        sync_run_id: str,
        trail: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm each ingested record is traceable source → Nango → graph.

        Each trail hop should carry `source_id` plus `nango_id` (or
        `record_id`) and `audit_id`. Optional `graph_id`.

        Args:
            connector_id: Nango connection id.
            sync_run_id: Sync run these hops belong to.
            trail: Per-record lineage dicts.

        Returns:
            ValidationResult. score is complete-hops / hops.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(connector_id, "connector_id", "audit_continuity")
            self._require_text(sync_run_id, "sync_run_id", "audit_continuity")
            if not trail:
                raise IngestionValidatorError(
                    "audit_continuity requires a non-empty trail",
                    evaluator_name=self.name,
                    metric="audit_continuity",
                )
            incomplete: list[str] = []
            for hop in trail:
                ident = str(hop.get("source_id") or hop.get("id") or "")
                nango = hop.get("nango_id") or hop.get("record_id")
                entry = hop.get("audit_id")
                if not ident or not nango or not entry:
                    incomplete.append(ident or "unknown")
            score = _clamp((len(trail) - len(incomplete)) / len(trail))
            result = ValidationResult(
                passed=not incomplete,
                score=score,
                evaluator_name=self.name,
                metric="audit_continuity",
                threshold=1.0,
                details={
                    "connector_id": connector_id,
                    "sync_run_id": sync_run_id,
                    "hop_count": len(trail),
                    "incomplete": incomplete,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "audit_continuity", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_file_ingestion(
        self,
        *,
        filename: str,
        extracted: list[dict[str, Any]] | None = None,
        expected_record_count: int | None = None,
        sample_fields: list[str] | None = None,
        encoding_error: bool = False,
        crashed: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a file ingest produced rows without crashing.

        Unit tests pass already-extracted `extracted` rows. Live parsers
        are out of scope for Phase 8 — this checks the ingestion contract.

        Args:
            filename: Source file name (extension is validated).
            extracted: Rows the pipeline produced.
            expected_record_count: Optional expected row count.
            sample_fields: Fields that must be present.
            encoding_error: Pipeline reported a decode failure.
            crashed: Pipeline aborted.

        Returns:
            ValidationResult. empty extract without an encoding alert fails.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(filename, "filename", "file_ingestion")
            suffix = Path(filename).suffix.lower()
            supported = suffix in FILE_EXTENSIONS or suffix == ""
            rows = list(extracted or [])
            count_ok = (
                expected_record_count is None or len(rows) == expected_record_count
            )
            missing_fields = [
                field
                for field in (sample_fields or [])
                if not rows or any(not field_present(row, field) for row in rows)
            ]
            empty = not rows and not encoding_error
            passed = (
                supported
                and not crashed
                and not empty
                and count_ok
                and not missing_fields
                and not encoding_error
            )
            # Encoding errors that are *reported* (not crashed) still fail
            # completeness but are not silent — recorded in details.
            result = ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluator_name=self.name,
                metric="file_ingestion",
                threshold=1.0,
                details={
                    "filename": filename,
                    "extension": suffix,
                    "supported": supported,
                    "actual_count": len(rows),
                    "expected_count": expected_record_count,
                    "missing_fields": missing_fields,
                    "encoding_error": encoding_error,
                    "crashed": crashed,
                    "empty": empty,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "file_ingestion", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(self, **kwargs: Any) -> list[ValidationResult]:
        """Run every ingestion check that has enough arguments.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results: list[ValidationResult] = []
        connector_id = str(kwargs.get("connector_id") or "")
        if connector_id and kwargs.get("expected_record_count") is not None:
            results.append(
                await self.validate_connector_reliability(
                    connector_id=connector_id,
                    expected_record_count=int(kwargs["expected_record_count"]),
                    sample_fields=list(kwargs.get("sample_fields") or []),
                    records=kwargs.get("records"),
                    model=str(kwargs.get("model") or ""),
                )
            )
        if connector_id and kwargs.get("injected_failure"):
            results.append(
                await self.validate_silent_failure(
                    connector_id=connector_id,
                    injected_failure=str(kwargs["injected_failure"]),
                    records=kwargs.get("records"),
                    error=kwargs.get("error"),
                    status_code=kwargs.get("status_code"),
                )
            )
        if connector_id and kwargs.get("expected_ids"):
            results.append(
                await self.validate_completeness(
                    connector_id=connector_id,
                    expected_ids=list(kwargs["expected_ids"]),
                    sample_fields=list(kwargs.get("sample_fields") or []),
                    records=kwargs.get("records"),
                )
            )
        if kwargs.get("original_schema") and kwargs.get("drifted_schema"):
            results.append(
                await self.validate_schema_drift(
                    connector_id=connector_id or "unknown",
                    original_schema=dict(kwargs["original_schema"]),
                    drifted_schema=dict(kwargs["drifted_schema"]),
                    handling=str(kwargs.get("handling") or ""),
                    required_fields=list(kwargs.get("required_fields") or []),
                    crashed=bool(kwargs.get("crashed")),
                )
            )
        if kwargs.get("filename"):
            results.append(
                await self.validate_file_ingestion(
                    filename=str(kwargs["filename"]),
                    extracted=kwargs.get("extracted"),
                    expected_record_count=kwargs.get("expected_record_count"),
                    sample_fields=list(kwargs.get("sample_fields") or []),
                    encoding_error=bool(kwargs.get("encoding_error")),
                    crashed=bool(kwargs.get("crashed")),
                )
            )
        return results

    async def _load_records(
        self,
        records: list[dict[str, Any]] | None,
        connector_id: str,
        model: str,
        provider_config_key: str,
    ) -> list[dict[str, Any]]:
        """Return caller-supplied records or fetch them from Nango.

        Args:
            records: In-memory rows, or None to call the integration.
            connector_id: Nango connection id.
            model: Synced model name.
            provider_config_key: Optional integration key.

        Returns:
            Record dicts.
        """
        if records is not None:
            return list(records)
        if not model:
            raise IngestionValidatorError(
                "live record fetch requires model",
                evaluator_name=self.name,
                metric="connector_reliability",
            )
        return await self._integration.list_records(
            connection_id=connector_id,
            model=model,
            provider_config_key=provider_config_key,
        )

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise IngestionValidatorError if `value` is empty or whitespace."""
        if not isinstance(value, str) or not value.strip():
            raise IngestionValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


NangoValidator = IngestionValidator
