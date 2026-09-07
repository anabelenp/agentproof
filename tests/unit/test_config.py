"""Unit tests for AgentProofConfig."""

from pathlib import Path

import pytest
from agentproof.core.config import AgentProofConfig
from pydantic import ValidationError


def test_default_thresholds():
    config = AgentProofConfig()
    assert config.default_relevance_threshold == 0.7
    assert config.faithfulness_threshold == 0.9
    assert config.executive_faithfulness_threshold == 0.95
    assert config.hallucination_threshold == 0.1
    assert config.contextual_recall_threshold == 0.7
    assert config.contextual_precision_threshold == 0.8
    assert config.toxicity_threshold == 0.1
    assert config.pii_redaction is True


def test_default_sla():
    config = AgentProofConfig()
    assert config.max_ttft_seconds == 2.0
    assert config.min_token_throughput == 20.0
    assert config.max_retrieval_latency_ms == 500.0
    assert config.max_fallback_seconds == 5.0
    assert config.routing_consistency_delta == 0.15
    assert config.max_db_write_latency_ms == 200.0
    assert config.ttl_tolerance_seconds == 5


def test_default_data_layer_urls(monkeypatch):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("POSTGRES_POOL_SIZE", raising=False)
    monkeypatch.delenv("REDIS_TTL_SECONDS", raising=False)
    config = AgentProofConfig()
    assert config.postgres_url == "postgresql://localhost:5432/agentproof_test"
    assert config.postgres_pool_size == 5
    assert config.redis_url == "redis://localhost:6379"
    assert config.redis_ttl_seconds == 3600


def test_default_judge_model():
    config = AgentProofConfig()
    assert config.judge_model == "claude-sonnet-4-6"


def test_default_retry():
    config = AgentProofConfig()
    assert config.max_retries == 3
    assert config.retry_base_delay == 1.0
    assert config.retry_max_delay == 60.0


def test_default_api_keys_are_none(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    config = AgentProofConfig()
    assert config.anthropic_api_key is None
    assert config.openai_api_key is None
    assert config.qdrant_api_key is None


def test_audit_log_dir_is_path():
    config = AgentProofConfig()
    assert isinstance(config.audit_log_dir, Path)


def test_threshold_override():
    config = AgentProofConfig(default_relevance_threshold=0.85)
    assert config.default_relevance_threshold == 0.85


def test_judge_model_override():
    config = AgentProofConfig(judge_model="claude-opus-4-7")
    assert config.judge_model == "claude-opus-4-7"


def test_threshold_too_high_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(default_relevance_threshold=1.5)


def test_threshold_negative_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(faithfulness_threshold=-0.1)


def test_ttft_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_ttft_seconds=0.0)


def test_retrieval_latency_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_retrieval_latency_ms=0.0)


def test_fallback_seconds_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_fallback_seconds=0.0)


def test_consistency_delta_above_one_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(routing_consistency_delta=1.5)


def test_db_write_latency_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_db_write_latency_ms=0.0)


def test_postgres_pool_size_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(postgres_pool_size=0)


def test_env_var_sets_postgres_and_redis_urls(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://ci:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://ci:6379/0")
    config = AgentProofConfig()
    assert config.postgres_url == "postgresql://ci:5432/test"
    assert config.redis_url == "redis://ci:6379/0"


def test_max_retries_zero_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_retries=0)


def test_max_retries_over_limit_raises():
    with pytest.raises(ValidationError):
        AgentProofConfig(max_retries=11)


def test_env_var_sets_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    config = AgentProofConfig()
    assert config.anthropic_api_key == "sk-test-123"


def test_env_var_sets_qdrant_url(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://remote-qdrant:6333")
    config = AgentProofConfig()
    assert config.qdrant_url == "http://remote-qdrant:6333"


def test_audit_log_dir_override(tmp_path):
    config = AgentProofConfig(audit_log_dir=tmp_path / "custom_logs")
    assert config.audit_log_dir == tmp_path / "custom_logs"
