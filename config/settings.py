"""Loads every setting from the .env file in one place."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder that contains "config/".
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read the .env file and put its values into the environment.
load_dotenv(PROJECT_ROOT / ".env")


def _get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    value = _get_str(name)
    return int(value) if value else default


def _get_bool(name: str, default: bool) -> bool:
    value = _get_str(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _get_path(name: str, default: str) -> Path:
    path = Path(_get_str(name, default))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str
    llm_model: str
    openai_api_key: str
    anthropic_api_key: str
    # Embeddings and vector DB
    embedding_model_name: str
    vector_db_path: Path
    collection_name: str
    # Data and retrieval
    subtitles_dir: Path
    top_k: int
    chunk_min_seconds: int
    chunk_max_seconds: int
    # Email
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from: str
    default_recipient_email: str
    email_dry_run: bool

    def smtp_is_configured(self) -> bool:
        """True if the minimum email settings are filled in."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)


def load_settings() -> Settings:
    return Settings(
        llm_provider=_get_str("LLM_PROVIDER", "openai"),
        llm_model=_get_str("LLM_MODEL"),
        openai_api_key=_get_str("OPENAI_API_KEY"),
        anthropic_api_key=_get_str("ANTHROPIC_API_KEY"),
        embedding_model_name=_get_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        vector_db_path=_get_path("VECTOR_DB_PATH", "data/chroma_db"),
        collection_name=_get_str("COLLECTION_NAME", "movie_subtitles"),
        subtitles_dir=_get_path("SUBTITLES_DIR", "data/subtitles"),
        top_k=_get_int("TOP_K", 5),
        chunk_min_seconds=_get_int("CHUNK_MIN_SECONDS", 30),
        chunk_max_seconds=_get_int("CHUNK_MAX_SECONDS", 90),
        smtp_host=_get_str("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=_get_int("SMTP_PORT", 587),
        smtp_user=_get_str("SMTP_USER"),
        smtp_password=_get_str("SMTP_PASSWORD"),
        email_from=_get_str("EMAIL_FROM"),
        default_recipient_email=_get_str("DEFAULT_RECIPIENT_EMAIL"),
        email_dry_run=_get_bool("EMAIL_DRY_RUN", True),
    )


# Other files do:  from config.settings import settings
settings = load_settings()