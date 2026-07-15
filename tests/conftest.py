"""Shared pytest fixtures for AgentProof unit and integration tests."""

import pytest

from agentproof.core.audit import AuditLogger
from agentproof.core.config import AgentProofConfig


@pytest.fixture
def config() -> AgentProofConfig:
    """Base AgentProofConfig with test defaults (no .env required)."""
    return AgentProofConfig()


@pytest.fixture
def audit_logger(tmp_path) -> AuditLogger:
    """AuditLogger writing to a pytest-managed temporary directory."""
    return AuditLogger(log_dir=tmp_path / "audit_logs")
