"""Configuration module for the trading assistant.

This module centralizes the reading of environment variables and provides a
`Settings` object that other modules can import.  It uses Pydantic's
`BaseSettings` to automatically read values from a `.env` file when present.

For production deployments you should set these environment variables in your
hosting provider's secrets manager rather than storing them in version control.
"""

from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    polygon_api_key: str | None = Field(None, env="POLYGON_API_KEY")
    db_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DB_URL", "DATABASE_URL"),
    )
    backend_api_key: str | None = Field(None, env="BACKEND_API_KEY")
    tradier_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TRADIER_API_TOKEN", "TRADIER_SANDBOX_TOKEN"),
    )
    tradier_account_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TRADIER_ACCOUNT_ID", "TRADIER_SANDBOX_ACCOUNT"),
    )
    tradier_base_url: str = Field(
        default="https://sandbox.tradier.com",
        validation_alias=AliasChoices("TRADIER_BASE_URL", "TRADIER_SANDBOX_BASE_URL"),
    )
    chart_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("BASE_URL", "CHART_BASE_URL"),
    )
    enrichment_service_url: str | None = Field(
        default="http://localhost:8081",
        env="ENRICH_SERVICE_URL",
    )
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    self_base_url: str | None = Field(None, env="SELF_API_BASE_URL")
    public_base_url: str | None = Field(None, env="PUBLIC_BASE_URL")
    index_sniper_mode: bool = Field(False, env="INDEX_SNIPER_MODE")
    ff_chart_canonical_v1: bool = Field(False, env="FF_CHART_CANONICAL_V1")
    ff_layers_endpoint: bool = Field(False, env="FF_LAYERS_ENDPOINT")
    ff_options_always: bool = Field(False, env="FF_OPTIONS_ALWAYS")

@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings.

    Pydantic caches the parsed environment variables so that repeated calls
    throughout the application are inexpensive.
    """

    return Settings()
