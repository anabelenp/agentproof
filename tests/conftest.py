"""Shared pytest fixtures for AgentProof unit and integration tests."""

from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def mock_anthropic():
    """Patch AsyncAnthropic so unit tests never open a network connection."""
    with patch("agentproof.integrations.anthropic.AsyncAnthropic") as mocked:
        client = AsyncMock()
        mocked.return_value = client
        block = MagicMock()
        block.text = "mocked completion"
        message = MagicMock()
        message.content = [block]
        client.messages.create = AsyncMock(return_value=message)
        client.close = AsyncMock()
        yield mocked


@pytest.fixture
def mock_qdrant():
    """Patch QdrantClient so unit tests never open a network connection."""
    with patch("agentproof.integrations.qdrant.QdrantClient") as mocked:
        client = MagicMock()
        mocked.return_value = client
        hit = MagicMock()
        hit.id = "doc-1"
        hit.score = 0.91
        hit.payload = {"text": "Refunds are available within 30 days."}
        response = MagicMock()
        response.points = [hit]
        client.query_points.return_value = response
        count_result = MagicMock()
        count_result.count = 1
        client.count.return_value = count_result
        client.retrieve.return_value = [hit]
        client.close = MagicMock()
        yield mocked


@pytest.fixture
def mock_litellm():
    """Patch litellm.acompletion so unit tests never open a network connection."""
    with patch(
        "agentproof.integrations.litellm.litellm.acompletion", new_callable=AsyncMock
    ) as mocked:
        message = MagicMock()
        message.content = "mocked routing response"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        response.model = "claude-sonnet-4-6"
        mocked.return_value = response
        yield mocked


@pytest.fixture
def mock_postgres():
    """Patch asyncpg.create_pool so unit tests never open a network connection."""
    with patch(
        "agentproof.integrations.postgres.asyncpg.create_pool", new_callable=AsyncMock
    ) as mocked:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx)

        pool = MagicMock()
        acquire = MagicMock()
        acquire.__aenter__ = AsyncMock(return_value=conn)
        acquire.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acquire)
        pool.close = AsyncMock()
        mocked.return_value = pool
        yield mocked


@pytest.fixture
def mock_redis():
    """Patch redis.from_url so unit tests never open a network connection."""
    with patch("agentproof.integrations.redis.redis_async.from_url") as mocked:
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.ttl = AsyncMock(return_value=3600)
        client.exists = AsyncMock(return_value=1)
        client.hgetall = AsyncMock(return_value={})
        client.hset = AsyncMock(return_value=1)
        client.aclose = AsyncMock()
        client.close = AsyncMock()
        mocked.return_value = client
        yield mocked


@pytest.fixture
def mock_deepeval_metric():
    """Patch AnswerRelevancyMetric with a deterministic async score."""
    with patch("agentproof.evaluators.llm.AnswerRelevancyMetric") as mocked:
        instance = mocked.return_value
        instance.score = 0.85
        instance.reason = "mocked reason"
        instance.success = True
        instance.a_measure = AsyncMock(return_value=0.85)
        yield mocked
