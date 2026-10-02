"""Application settings, read from environment variables and an optional `.env` file.

Every tunable lives here so the rest of the code never touches `os.environ`.
Environment variables take precedence over `.env`; see `.env.example`.
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

NO_KEY = "not-set"


class Settings(BaseSettings):
    """Runtime configuration of the assistant."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM endpoint: any OpenAI-compatible API. The defaults target Groq.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr = SecretStr(NO_KEY)
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0, description="Retries on HTTP 429 / 5xx.")
    llm_retry_base_s: float = Field(default=2.0, ge=0, description="First backoff delay.")

    # Agent loop
    max_steps: int = Field(default=5, ge=1, description="Max LLM round-trips per question.")

    # Rate limit on /ask, for public demos running on a shared LLM quota (0 = no limit).
    ask_rate_limit: int = Field(default=0, ge=0, description="Questions per client per window.")
    ask_global_limit: int = Field(default=0, ge=0, description="Questions per window, in total.")
    ask_rate_window_s: int = Field(default=3600, gt=0)
    # Behind a reverse proxy the client address is read from X-Forwarded-For.
    trust_forwarded_for: bool = False

    # Models: bundled samples, and limits on what visitors may upload (kept in memory).
    samples_dir: Path = Path("data/samples")
    max_upload_mb: float = Field(default=10.0, gt=0)
    max_faces: int = Field(default=200_000, ge=1, description="Largest mesh accepted.")
    max_uploads_per_session: int = Field(default=2, ge=1)
    max_sessions: int = Field(default=20, ge=1, description="Sessions kept; oldest dropped.")

    log_level: str = "INFO"

    @property
    def llm_key_is_placeholder(self) -> bool:
        """True when no API key was provided (only valid for a local server)."""
        return self.llm_api_key.get_secret_value() == NO_KEY

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit in bytes."""
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (loaded once)."""
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    """Set up console logging for the application and its scripts."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # One line per HTTP request to the LLM is too noisy at INFO level.
    for name in ("httpx", "httpx2"):  # the OpenAI SDK logs through either, by version
        logging.getLogger(name).setLevel(logging.WARNING)
