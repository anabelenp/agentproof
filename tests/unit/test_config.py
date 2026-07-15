"""Unit tests for AgentProofConfig."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentproof.core.config import AgentProofConfig


def test_default_thresholds():
    config = AgentProofConfig()
    assert config.default_relevance_threshold == 0.7
    assert config.faithfulness_threshold == 0.9
    assert config.executive_faithfulness_threshold == 0.95
    assert config.hallucination_threshold == 0.1
    assert config.contextual_recall_threshold == 0.7
    assert config.contextual_precision_threshold == 0.8


def test_default_sla():
    config = AgentProofConfig()
    assert config.max_ttft_seconds == 2.0
    assert config.min_token_throughput == 20.0


def test_default_judge_model():
    config = AgentProofConfig()
    assert config.judge_model == "claude-sonnet-4-6"


def test_default_retry():
    config = AgentProofConfig()
    assert config.max_retries == 3
    assert config.retry_base_delay == 1.0
    assert config.retry_max_delay == 60.0


def test_default_api_keys_are_none():
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
