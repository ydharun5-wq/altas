"""Environment-backed settings for the local ATLAS web application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./atlas.db"
    environment: str = "development"
    bootstrap_token: str | None = None
    cookie_secure: bool = False
    session_ttl_hours: int = 12
    login_max_failures: int = 5
    login_window_seconds: int = 900
    quote_max_age_seconds: int = 300
    allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )
    enable_api_docs: bool = False
    market_data_dir: Path | None = None
    market_data_source: str | None = None
    market_data_is_synthetic: bool | None = None

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    @field_validator("session_ttl_hours", "login_max_failures", "login_window_seconds", "quote_max_age_seconds")
    @classmethod
    def positive_limits(cls, value: int) -> int:
        if value < 1:
            raise ValueError("security limits must be positive")
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            return tuple(part.strip().rstrip("/") for part in value.split(",") if part.strip())
        return tuple(origin.rstrip("/") for origin in value)

    @field_validator("environment")
    @classmethod
    def secure_production_cookies(cls, value: str) -> str:
        if value not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")
        return value

    def validate_runtime_security(self) -> None:
        if self.environment == "production" and not self.cookie_secure:
            raise ValueError("ATLAS_COOKIE_SECURE must be true in production")
        if self.bootstrap_token is not None and len(self.bootstrap_token) < 32:
            raise ValueError("ATLAS_BOOTSTRAP_TOKEN must contain at least 32 characters")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime_security()
    return settings
