"""Unit tests for AuditLogger and AuditEntry."""

import json
import uuid

import pytest

from agentproof.core.audit import AuditEntry, AuditLogger, new_audit_id
from agentproof.core.errors import AuditError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def logger(tmp_path) -> AuditLogger:
    return AuditLogger(log_dir=tmp_path / "audit_logs")


@pytest.fixture
def entry() -> AuditEntry:
    return AuditEntry(
        audit_id=new_audit_id(),
        evaluator="LLMEvaluator",
        metric="faithfulness",
        score=0.94,
        threshold=0.90,
        passed=True,
        model="claude-sonnet-4-6",
        latency_ms=342.1,
        details={"reason": "output grounded in context"},
    )


# ── File creation ─────────────────────────────────────────────────────────────


async def test_log_creates_directory_and_file(logger, entry):
    await logger.log(entry)
    assert logger._today_log_path().exists()


async def test_log_dir_created_automatically(tmp_path, entry):
    deep_path = tmp_path / "a" / "b" / "c" / "audit_logs"
    logger = AuditLogger(log_dir=deep_path)
    await logger.log(entry)
    assert deep_path.exists()


# ── JSONL format ──────────────────────────────────────────────────────────────


async def test_log_writes_valid_jsonl(logger, entry):
    await logger.log(entry)
    lines = logger._today_log_path().read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["audit_id"] == entry.audit_id
    assert data["evaluator"] == "LLMEvaluator"
    assert data["metric"] == "faithfulness"
    assert data["score"] == 0.94
    assert data["passed"] is True


async def test_log_includes_timestamp(logger, entry):
    await logger.log(entry)
    data = json.loads(logger._today_log_path().read_text().strip())
    assert data["timestamp"] != ""
    assert "T" in data["timestamp"]  # ISO-8601


async def test_log_sets_timestamp_on_entry(logger, entry):
    assert entry.timestamp == ""
    await logger.log(entry)
    assert entry.timestamp != ""


# ── SHA-256 hash ──────────────────────────────────────────────────────────────


async def test_log_includes_sha256_hash(logger, entry):
    await logger.log(entry)
    data = json.loads(logger._today_log_path().read_text().strip())
    assert "entry_hash" in data
    assert data["entry_hash"].startswith("sha256:")
    assert len(data["entry_hash"]) == len("sha256:") + 64


async def test_log_sets_hash_on_entry(logger, entry):
    assert entry.entry_hash == ""
    await logger.log(entry)
    assert entry.entry_hash.startswith("sha256:")


# ── Multiple entries ──────────────────────────────────────────────────────────


async def test_multiple_entries_appended_as_separate_lines(logger):
    entries = [
        AuditEntry(
            audit_id=new_audit_id(),
            evaluator="LLMEvaluator",
            metric="relevance",
            score=0.8,
            threshold=0.7,
            passed=True,
            model="claude-sonnet-4-6",
            latency_ms=100.0,
        )
        for _ in range(5)
    ]
    for e in entries:
        await logger.log(e)

    lines = [
        l for l in logger._today_log_path().read_text().strip().split("\n") if l
    ]
    assert len(lines) == 5


async def test_each_line_is_valid_json(logger):
    for _ in range(3):
        e = AuditEntry(
            audit_id=new_audit_id(),
            evaluator="TestEval",
            metric="test",
            score=0.5,
            threshold=0.5,
            passed=True,
            model="",
            latency_ms=10.0,
        )
        await logger.log(e)

    for line in logger._today_log_path().read_text().strip().split("\n"):
        json.loads(line)  # should not raise


# ── Integrity verification ────────────────────────────────────────────────────


async def test_verify_integrity_valid_entry(logger, entry):
    await logger.log(entry)
    assert await logger.verify_integrity(entry.audit_id) is True


async def test_verify_integrity_nonexistent_id(logger):
    assert await logger.verify_integrity("does-not-exist") is False


async def test_verify_integrity_no_log_file(tmp_path):
    logger = AuditLogger(log_dir=tmp_path / "missing")
    assert await logger.verify_integrity("any-id") is False


async def test_verify_integrity_tampered_score(logger, entry):
    await logger.log(entry)
    log_path = logger._today_log_path()

    data = json.loads(log_path.read_text().strip())
    data["score"] = 0.0  # tamper
    log_path.write_text(json.dumps(data) + "\n")

    assert await logger.verify_integrity(entry.audit_id) is False


async def test_verify_integrity_tampered_passed_flag(logger, entry):
    await logger.log(entry)
    log_path = logger._today_log_path()

    data = json.loads(log_path.read_text().strip())
    data["passed"] = False  # tamper
    log_path.write_text(json.dumps(data) + "\n")

    assert await logger.verify_integrity(entry.audit_id) is False


# ── new_audit_id ──────────────────────────────────────────────────────────────


async def test_read_entries_empty_dir(tmp_path):
    logger = AuditLogger(log_dir=tmp_path / "nope")
    assert await logger.read_entries() == []


async def test_read_entries_returns_logged_rows(logger, entry):
    await logger.log(entry)
    rows = await logger.read_entries()
    assert len(rows) == 1
    assert rows[0]["audit_id"] == entry.audit_id
    assert rows[0]["metric"] == "faithfulness"


async def test_verify_entry_data_accepts_logged_row(logger, entry):
    await logger.log(entry)
    rows = await logger.read_entries()
    assert logger.verify_entry_data(rows[0]) is True


async def test_verify_entry_data_rejects_missing_hash():
    logger = AuditLogger(log_dir=".")
    assert logger.verify_entry_data({"audit_id": "x", "score": 1}) is False


async def test_verify_entry_data_rejects_tampered_row(logger, entry):
    await logger.log(entry)
    rows = await logger.read_entries()
    rows[0]["score"] = 0.01
    assert logger.verify_entry_data(rows[0]) is False


def test_new_audit_id_is_valid_uuid():
    aid = new_audit_id()
    parsed = uuid.UUID(aid)
    assert str(parsed) == aid


def test_new_audit_id_is_unique():
    ids = {new_audit_id() for _ in range(100)}
    assert len(ids) == 100


# ── AuditError propagation ────────────────────────────────────────────────────


async def test_audit_error_raised_on_unwritable_path(entry):
    logger = AuditLogger(log_dir="/root/cannot_write_here_ever")
    with pytest.raises(AuditError):
        await logger.log(entry)
