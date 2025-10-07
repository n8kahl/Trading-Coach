"""Configuration module for the trading assistant.

This module centralizes the reading of environment variables and provides a
`Settings` object that other modules can import.  It uses Pydantic's
`BaseSettings` to automatically read values from a `.env` file when present.

For production deployments you should set these environment variables in your
hosting provider's secrets manager rather than storing them in version control.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    workflow_id: str = Field(..., env="WORKFLOW_ID")
    chatkit_api_key: str | None = Field(None, env="CHATKIT_API_KEY")
    chatkit_user_id: str = Field("demo-user", env="CHATKIT_USER_ID")
    db_url: str | None = Field(None, env="DB_URL")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings.

    Pydantic caches the parsed environment variables so that repeated calls
    throughout the application are inexpensive.
    """

    return Settings()
