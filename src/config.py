"""Configuration module for the trading assistant.

This module centralizes the reading of environment variables and provides a
`Settings` object that other modules can import.  It uses Pydantic's
`BaseSettings` to automatically read values from a `.env` file when present.

For production deployments you should set these environment variables in your
hosting provider's secrets manager rather than storing them in version control.
"""

from functools import lru_cache
from pydantic import AliasChoices, Field, field_validator
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
        validation_alias=AliasChoices("ENRICH_SERVICE_URL", "enrichment_service_url"),
    )
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    self_base_url: str | None = Field(None, env="SELF_API_BASE_URL")
    public_base_url: str | None = Field(None, env="PUBLIC_BASE_URL")
    index_sniper_mode: bool = Field(False, env="INDEX_SNIPER_MODE")
    ff_chart_canonical_v1: bool = Field(False, env="FF_CHART_CANONICAL_V1")
    ff_layers_endpoint: bool = Field(False, env="FF_LAYERS_ENDPOINT")
    ff_options_always: bool = Field(False, env="FF_OPTIONS_ALWAYS")
    gpt_market_routing_enabled: bool = Field(True, env="GPT_MARKET_ROUTING_ENABLED")
    gpt_backend_v2_enabled: bool = Field(True, env="GPT_BACKEND_V2_ENABLED")
    ft_no_fallback_trades: bool = Field(False, env="FT_NO_FALLBACK_TRADES")
    ft_disable_delta_filters: bool = Field(False, env="FT_DISABLE_DELTA_FILTERS")
    ft_disable_spread_guardrail: bool = Field(False, env="FT_DISABLE_SPREAD_GUARDRAIL")
    ft_max_spread_pct: float = Field(8.0, env="FT_MAX_SPREAD_PCT")
    ft_min_oi: int = Field(300, env="FT_MIN_OI")
    ft_max_drift_bps: float = Field(30.0, env="FT_MAX_DRIFT_BPS")
    ft_em_factor: float = Field(1.3, env="FT_EM_FACTOR")
    ft_allowed_hosts: list[str] = Field(
        default_factory=lambda: ["https://trading-coach-production.up.railway.app"],
        env="FT_ALLOWED_HOSTS",
    )
    ft_event_blocked_styles: list[str] = Field(default_factory=list, env="FT_EVENT_BLOCKED_STYLES")
    calibration_data_path: str | None = Field(None, env="CALIBRATION_DATA_PATH")

    @field_validator("ft_allowed_hosts", "ft_event_blocked_styles", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value

@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings.

    Pydantic caches the parsed environment variables so that repeated calls
    throughout the application are inexpensive.
    """

    return Settings()


UNIFIED_SNAPSHOT_ENABLED: bool = True
SNAPSHOT_MAX_CONCURRENCY: int = 8
SNAPSHOT_INTERVAL: str = "1m"
SNAPSHOT_LOOKBACK: int = 300

_SCALP_GATES = {
    "hard_pct_cap": 0.0012,
    "pref_pct": 0.0008,
    "hard_atr_cap": 0.35,
    "pref_atr": 0.24,
    "hard_bars_cap": 2,
    "pref_bars": 1,
}

STYLE_GATES: dict[str, dict[str, float]] = {
    "scalp": dict(_SCALP_GATES),
    "0dte": dict(_SCALP_GATES),
    "intraday": {
        "hard_pct_cap": 0.0025,
        "pref_pct": 0.0018,
        "hard_atr_cap": 0.70,
        "pref_atr": 0.50,
        "hard_bars_cap": 4,
        "pref_bars": 3,
    },
    "swing": {
        "hard_pct_cap": 0.0042,
        "pref_pct": 0.0030,
        "hard_atr_cap": 1.00,
        "pref_atr": 0.80,
        "hard_bars_cap": 7,
        "pref_bars": 5,
    },
    "leaps": {
        "hard_pct_cap": 0.0065,
        "pref_pct": 0.0045,
        "hard_atr_cap": 1.40,
        "pref_atr": 1.20,
        "hard_bars_cap": 10,
        "pref_bars": 8,
    },
}

LENIENT_GATES_DEFAULT: bool = True
LENIENCY_PENALTY_SCALE: float = 0.25
VOLATILITY_RELAX_VIX: float = 25.0
