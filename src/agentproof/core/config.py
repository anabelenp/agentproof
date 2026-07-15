"""AgentProofConfig — central configuration for the evaluation framework.

Merges environment variables (.env file + shell env) with sensible defaults.
All secrets come from env vars only — never hardcoded.
Raises pydantic ValidationError on invalid threshold values at startup.

Usage:
    config = AgentProofConfig()                              # reads from .env + env
    config = AgentProofConfig(judge_model="claude-opus-4-7") # override a field
    config = AgentProofConfig(_env_file="custom.env")        # custom env file
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentProofConfig(BaseSettings):
    """Pydantic BaseSettings that loads from environment + .env file.

    Field names map to uppercase env vars automatically:
      anthropic_api_key → ANTHROPIC_API_KEY
      qdrant_url        → QDRANT_URL
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API keys (from env only, default None so offline testing works) ──────
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    qdrant_api_key: str | None = None

    # ── Model ─────────────────────────────────────────────────────────────────
    judge_model: str = "claude-sonnet-4-6"

    # ── External services ──────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_log_dir: Path = Path("./audit_logs")

    # ── Quality thresholds (all validated 0.0–1.0) ────────────────────────────
    default_relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    executive_faithfulness_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    contextual_recall_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    contextual_precision_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # ── SLA ───────────────────────────────────────────────────────────────────
    max_ttft_seconds: float = Field(default=2.0, gt=0)
    min_token_throughput: float = Field(default=20.0, gt=0)

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0)
    retry_max_delay: float = Field(default=60.0, gt=0)
