"""Unit tests for NangoIntegration and IngestionValidator. HTTP is mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, IntegrationError
from agentproof.core.runner import TestRunner
from agentproof.integrations.nango import NangoIntegration, normalize_records
from agentproof.validators.ingestion import (
    IngestionValidator,
    classify_drift,
    field_present,
    record_id,
)


def _response(status: int = 200, payload: object | None = None):
    response = MagicMock()
    response.status_code = status
    response.content = b"{}" if payload is not None else b""
    response.json = MagicMock(return_value=payload if payload is not None else {})
    return response


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.request = AsyncMock(return_value=_response(200, {"records": []}))
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def integration(mock_client) -> NangoIntegration:
    return NangoIntegration(AgentProofConfig(), client=mock_client)


@pytest.fixture
def mock_audit() -> AuditLogger:
    audit = MagicMock(spec=AuditLogger)
    audit.log = AsyncMock()
    return audit


@pytest.fixture
def validator(mock_audit, integration) -> IngestionValidator:
    return IngestionValidator(AgentProofConfig(), mock_audit, integration=integration)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_normalize_records_list_and_wrapper():
    assert normalize_records([{"id": "1"}]) == [{"id": "1"}]
    assert normalize_records({"records": [{"id": "2"}]}) == [{"id": "2"}]
    assert normalize_records({"data": [{"id": "3"}]}) == [{"id": "3"}]
    assert normalize_records(None) == []


def test_record_id_and_field_present():
    assert record_id({"id": "a"}) == "a"
    assert record_id({"source_id": "b"}) == "b"
    assert field_present({"email": "x"}, "email") is True
    assert field_present({"email": None}, "email") is False
    assert field_present({}, "email") is False


def test_classify_drift():
    drift = classify_drift(
        {"id": "1", "email": "a", "name": "n"},
        {"id": 1, "name": "n", "phone": "x"},
    )
    assert drift["added"] == ["phone"]
    assert drift["removed"] == ["email"]
    assert drift["type_changes"] == ["id"]


# ── NangoIntegration ──────────────────────────────────────────────────────────


def test_init_sets_bearer_header():
    with patch("agentproof.integrations.nango.httpx.AsyncClient") as cls:
        cls.return_value = MagicMock()
        NangoIntegration(
            AgentProofConfig(
                nango_api_key="nango-secret",
                nango_base_url="https://api.nango.dev",
            )
        )
    kwargs = cls.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer nango-secret"
    assert kwargs["base_url"] == "https://api.nango.dev"


async def test_list_records(integration, mock_client):
    mock_client.request = AsyncMock(
        return_value=_response(200, {"records": [{"id": "c1", "email": "a@b.c"}]})
    )
    rows = await integration.list_records(connection_id="sf-1", model="Contact")
    assert rows[0]["id"] == "c1"
    args, kwargs = mock_client.request.call_args
    assert args[0] == "GET"
    assert args[1] == "/records"
    assert kwargs["params"]["connectionId"] == "sf-1"
    assert kwargs["params"]["model"] == "Contact"


async def test_list_records_empty_connection_raises(integration):
    with pytest.raises(IntegrationError, match="connection_id"):
        await integration.list_records(connection_id="  ", model="Contact")


async def test_trigger_sync(integration, mock_client):
    mock_client.request = AsyncMock(return_value=_response(200, {"ok": True}))
    body = await integration.trigger_sync(connection_id="sf-1", syncs=["contacts"])
    assert body["ok"] is True
    assert mock_client.request.call_args.args[0] == "POST"


async def test_get_connection(integration, mock_client):
    mock_client.request = AsyncMock(
        return_value=_response(200, {"connection_id": "sf-1", "refreshed": True})
    )
    conn = await integration.get_connection("sf-1")
    assert conn["refreshed"] is True


async def test_request_retries_429(integration, mock_client):
    mock_client.request = AsyncMock(
        side_effect=[
            _response(429, {"error": "slow down"}),
            _response(200, {"records": []}),
        ]
    )
    with patch("agentproof.core.retry.asyncio.sleep", new_callable=AsyncMock):
        payload = await integration.request("GET", "/records")
    assert payload == {"records": []}
    assert mock_client.request.await_count == 2


async def test_request_4xx_is_integration_error(integration, mock_client):
    mock_client.request = AsyncMock(return_value=_response(401, {"error": "nope"}))
    with pytest.raises(IntegrationError, match="401"):
        await integration.request("GET", "/connection/x")


async def test_request_empty_path_raises(integration):
    with pytest.raises(IntegrationError, match="path"):
        await integration.request("GET", "  ")


async def test_close_does_not_close_injected_client(integration, mock_client):
    await integration.close()
    mock_client.aclose.assert_not_called()


# ── IngestionValidator identity ───────────────────────────────────────────────


def test_name(validator):
    assert validator.name == "IngestionValidator"


async def test_unknown_metric(validator, mock_audit):
    result = await validator.evaluate(metric="webhooks")
    assert "Unknown metric" in (result.error or "")
    mock_audit.log.assert_awaited_once()


# ── connector_reliability ─────────────────────────────────────────────────────


async def test_reliability_pass(validator):
    result = await validator.validate_connector_reliability(
        connector_id="sf-1",
        expected_record_count=2,
        sample_fields=["id", "email"],
        records=[
            {"id": "1", "email": "a@b.c"},
            {"id": "2", "email": "c@d.e"},
        ],
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_reliability_count_mismatch(validator):
    result = await validator.validate_connector_reliability(
        connector_id="sf-1",
        expected_record_count=3,
        sample_fields=[],
        records=[{"id": "1"}],
    )
    assert result.passed is False
    assert result.details["actual_count"] == 1
    assert result.error is None


async def test_reliability_missing_sample_field(validator):
    result = await validator.validate_connector_reliability(
        connector_id="sf-1",
        expected_record_count=1,
        sample_fields=["id", "email"],
        records=[{"id": "1"}],
    )
    assert result.passed is False
    assert result.details["missing_fields"] == ["email"]


# ── silent_failure ────────────────────────────────────────────────────────────


async def test_silent_failure_surfaced(validator):
    result = await validator.validate_silent_failure(
        connector_id="sf-1",
        injected_failure="drop_records",
        records=[],
        error="sync aborted: dropped batch",
    )
    assert result.passed is True
    assert result.details["silent_failure"] is False


async def test_silent_failure_swallowed(validator):
    result = await validator.validate_silent_failure(
        connector_id="sf-1",
        injected_failure="drop_records",
        records=[{"id": "1"}],
        error=None,
    )
    assert result.passed is False
    assert result.details["silent_failure"] is True
    assert result.error is None


async def test_silent_failure_auth_expire_401(validator):
    result = await validator.validate_silent_failure(
        connector_id="sf-1",
        injected_failure="auth_expire",
        status_code=401,
    )
    assert result.passed is True


async def test_silent_failure_bad_kind_is_error(validator, mock_audit):
    result = await validator.validate_silent_failure(
        connector_id="sf-1", injected_failure="explode"
    )
    assert result.error is not None
    mock_audit.log.assert_awaited_once()


# ── completeness ──────────────────────────────────────────────────────────────


async def test_completeness_pass(validator):
    result = await validator.validate_completeness(
        connector_id="sf-1",
        expected_ids=["a", "b"],
        sample_fields=["email"],
        records=[{"id": "a", "email": "a@x"}, {"id": "b", "email": "b@x"}],
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.threshold == 0.999


async def test_completeness_missing_id(validator):
    result = await validator.validate_completeness(
        connector_id="sf-1",
        expected_ids=["a", "b"],
        records=[{"id": "a"}],
    )
    assert result.passed is False
    assert result.details["missing_ids"] == ["b"]
    assert abs(result.score - 0.5) < 1e-9


# ── schema_drift — 10 scenarios ───────────────────────────────────────────────


def _schema_result(validator, **kwargs):
    return validator.validate_schema_drift(connector_id="sf-1", **kwargs)


async def test_drift_extra_optional_field_alert(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "email": "a"},
        drifted_schema={"id": "1", "email": "a", "phone": "x"},
        handling="alert",
        required_fields=["id", "email"],
    )
    assert result.passed is True
    assert result.details["added"] == ["phone"]


async def test_drift_extra_field_ignore(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1"},
        drifted_schema={"id": "1", "extra": "y"},
        handling="ignore",
        required_fields=["id"],
    )
    assert result.passed is True


async def test_drift_missing_optional_alert(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "note": "n"},
        drifted_schema={"id": "1"},
        handling="alert",
        required_fields=["id"],
    )
    assert result.passed is True
    assert result.details["removed"] == ["note"]


async def test_drift_missing_required_alert(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "email": "a"},
        drifted_schema={"id": "1"},
        handling="alert",
        required_fields=["id", "email"],
    )
    assert result.passed is True
    assert result.details["lost_required"] == ["email"]


async def test_drift_missing_required_ignore_is_silent(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "email": "a"},
        drifted_schema={"id": "1"},
        handling="ignore",
        required_fields=["email"],
    )
    assert result.passed is False
    assert result.details["silent"] is True
    assert result.error is None


async def test_drift_missing_required_unspecified_is_silent(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "email": "a"},
        drifted_schema={"id": "1"},
        handling="",
        required_fields=["email"],
    )
    assert result.passed is False
    assert result.details["silent"] is True


async def test_drift_crash_fails(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1"},
        drifted_schema={"id": "1", "phone": "x"},
        handling="crash",
        required_fields=["id"],
    )
    assert result.passed is False
    assert result.details["crashed"] is True


async def test_drift_type_change_alert(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "amount": "10"},
        drifted_schema={"id": "1", "amount": 10},
        handling="alert",
        required_fields=["amount"],
    )
    assert result.passed is True
    assert result.details["type_changes"] == ["amount"]


async def test_drift_type_change_ignore_required_fails(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "amount": "10"},
        drifted_schema={"id": "1", "amount": 10},
        handling="ignore",
        required_fields=["amount"],
    )
    assert result.passed is False
    assert result.details["typed_required"] == ["amount"]


async def test_drift_renamed_field_alert(validator):
    result = await _schema_result(
        validator,
        original_schema={"id": "1", "full_name": "Ada"},
        drifted_schema={"id": "1", "name": "Ada"},
        handling="alert",
        required_fields=["id"],
    )
    assert result.passed is True
    assert result.details["removed"] == ["full_name"]
    assert result.details["added"] == ["name"]


# ── auth_refresh / rate_limit / audit / files ─────────────────────────────────


async def test_auth_refresh_pass(validator):
    result = await validator.validate_auth_refresh(
        connector_id="sf-1",
        token_refreshed=True,
        data_loss=False,
        records=[{"id": "1"}],
    )
    assert result.passed is True


async def test_auth_refresh_data_loss(validator):
    result = await validator.validate_auth_refresh(
        connector_id="sf-1",
        token_refreshed=True,
        records=[],
    )
    assert result.passed is False
    assert result.details["data_loss"] is True


async def test_rate_limit_backoff_recovers(validator):
    result = await validator.validate_rate_limit(
        connector_id="sf-1",
        status_codes=[429, 429, 200],
        retried=True,
    )
    assert result.passed is True
    assert result.details["saw_429"] is True


async def test_rate_limit_429_without_retry_fails(validator):
    result = await validator.validate_rate_limit(
        connector_id="sf-1",
        status_codes=[429],
        retried=False,
    )
    assert result.passed is False


async def test_audit_continuity_pass(validator):
    result = await validator.validate_audit_continuity(
        connector_id="sf-1",
        sync_run_id="sync-9",
        trail=[
            {"source_id": "a", "nango_id": "n-a", "audit_id": "u1", "graph_id": "g1"},
            {"source_id": "b", "nango_id": "n-b", "audit_id": "u2"},
        ],
    )
    assert result.passed is True
    assert result.score == 1.0


async def test_audit_continuity_gap(validator):
    result = await validator.validate_audit_continuity(
        connector_id="sf-1",
        sync_run_id="sync-9",
        trail=[
            {"source_id": "a", "nango_id": "n-a", "audit_id": "u1"},
            {"source_id": "b"},
        ],
    )
    assert result.passed is False
    assert result.details["incomplete"] == ["b"]
    assert abs(result.score - 0.5) < 1e-9


async def test_file_ingestion_pdf_pass(validator):
    result = await validator.validate_file_ingestion(
        filename="policy.pdf",
        extracted=[{"id": "p1", "text": "Refunds within 30 days."}],
        expected_record_count=1,
        sample_fields=["text"],
    )
    assert result.passed is True


@pytest.mark.parametrize(
    "filename",
    ["notes.docx", "grid.xlsx", "deck.pptx", "rows.csv", "readme.md"],
)
async def test_file_ingestion_supported_types(validator, filename):
    result = await validator.validate_file_ingestion(
        filename=filename,
        extracted=[{"id": "1"}],
        expected_record_count=1,
    )
    assert result.passed is True
    assert result.details["supported"] is True


async def test_file_ingestion_empty_fails(validator):
    result = await validator.validate_file_ingestion(
        filename="empty.pdf", extracted=[]
    )
    assert result.passed is False
    assert result.details["empty"] is True
    assert result.error is None


async def test_file_ingestion_encoding_error(validator):
    result = await validator.validate_file_ingestion(
        filename="latin1.csv",
        extracted=[{"id": "1"}],
        encoding_error=True,
    )
    assert result.passed is False
    assert result.details["encoding_error"] is True


async def test_file_ingestion_crash(validator):
    result = await validator.validate_file_ingestion(
        filename="bad.pdf",
        extracted=[{"id": "1"}],
        crashed=True,
    )
    assert result.passed is False
    assert result.details["crashed"] is True


# ── evaluate_all / runner / AuditError ────────────────────────────────────────


async def test_evaluate_dispatches_reliability(validator):
    result = await validator.evaluate(
        metric="connector_reliability",
        connector_id="sf-1",
        expected_record_count=1,
        sample_fields=["id"],
        records=[{"id": "1"}],
    )
    assert result.metric == "connector_reliability"
    assert result.passed is True


async def test_evaluate_all_runs_provided_checks(validator):
    results = await validator.evaluate_all(
        connector_id="sf-1",
        expected_record_count=1,
        sample_fields=["id"],
        records=[{"id": "1"}],
        expected_ids=["1"],
        filename="a.pdf",
        extracted=[{"id": "1"}],
    )
    assert [r.metric for r in results] == [
        "connector_reliability",
        "completeness",
        "file_ingestion",
    ]


async def test_audit_error_propagates(validator, mock_audit):
    mock_audit.log = AsyncMock(side_effect=AuditError("disk full"))
    with pytest.raises(AuditError, match="disk full"):
        await validator.validate_connector_reliability(
            connector_id="sf-1",
            expected_record_count=0,
            sample_fields=[],
            records=[],
        )


async def test_runner_registers_ingestion(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "audit_logs")
    runner = TestRunner(config)
    integration = MagicMock()
    integration.list_records = AsyncMock(return_value=[])
    validator = IngestionValidator(config, runner.audit_logger, integration=integration)
    runner.register(
        validator,
        metric="file_ingestion",
        filename="policy.pdf",
        extracted=[{"id": "1"}],
        expected_record_count=1,
    )
    summary = await runner.run()
    assert summary.total == 1
    assert summary.passed == 1
