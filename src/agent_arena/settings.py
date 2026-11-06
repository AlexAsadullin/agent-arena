"""Typed settings: defaults from ``config/settings.yaml``, overridden by ``ARENA_*`` env variables.

Precedence (highest first): init kwargs, environment, ``.env``, the YAML file. Secrets (API keys,
the gateway admin key) are only ever read from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

SETTINGS_FILE_ENV = "ARENA_SETTINGS_FILE"
DEFAULT_SETTINGS_FILE = Path("config/settings.yaml")


class ServiceSettings(BaseModel):
    host: str
    port: int


class GatewaySettings(BaseModel):
    """Connection to the I9 MCP gateway (contract 6.6)."""

    def at(self, base_url: str) -> GatewaySettings:
        """The same settings pointed at another gateway: admin API at ``base_url``, MCP at the same path."""
        path = urlsplit(self.mcp_url).path or "/"
        base = base_url.rstrip("/")
        return self.model_copy(update={"admin_url": base, "mcp_url": base + path})

    mcp_url: str
    admin_url: str
    admin_auth_header: str
    admin_auth_scheme: str
    agent_auth_header: str
    agent_auth_scheme: str
    session_header: str
    mode: Literal["research", "paper", "live"]
    timeout_s: float
    clock_retries: int = Field(ge=0)
    admin_key: SecretStr | None = None
    agent_key: SecretStr | None = None


class LedgerSettings(BaseModel):
    """S6 research ledger (contract 6.1)."""

    base_url: str
    timeout_s: float
    offline_inbox: bool
    inbox_subdir: str
    source: str


class LeakageSettings(BaseModel):
    """S7 leakage probe (contract 6.5)."""

    base_url: str
    timeout_s: float
    languages: list[str]
    reports_subdir: str


class ProviderSettings(BaseModel):
    """One LLM provider endpoint. ``kind`` selects the adapter."""

    kind: Literal["openai_compat", "anthropic", "scripted"]
    base_url: str | None = None
    api_version: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    effort: str | None = None
    timeout_s: float = 60.0
    api_key: SecretStr | None = None


class ModelPrice(BaseModel):
    """USD per one million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class AccountingSettings(BaseModel):
    fallback_fee_bps: float = Field(ge=0)
    fill_price_source: Literal["order", "reference"]
    min_order_notional: float = Field(ge=0)


class ZeroFeeFundsSettings(BaseModel):
    """Instruments with their own rates on a venue (T-Bank's own funds on the MOEX venues)."""

    maker_fee: float = Field(ge=0)
    taker_fee: float = Field(ge=0)
    tickers: list[str]
    instrument_uids: list[str]
    # Relative to TRADEQ_DATA_DIR; an absent file is tolerated (config list only, with a warning).
    reference_file: str | None = None


class VenueFeeSettings(BaseModel):
    """Fee schedule of one venue: proportional rates on trade value per fill."""

    source: str
    maker_fee: float = Field(ge=0)
    taker_fee: float = Field(ge=0)
    # Fee per fill is rounded half-up to a multiple of this (0.01 = the kopeck); unset = no rounding.
    rounding: float | None = Field(default=None, gt=0)
    zero_fee_funds: ZeroFeeFundsSettings | None = None


class FeeSettings(BaseModel):
    venues: dict[str, VenueFeeSettings] = Field(default_factory=dict)


class LatencySettings(BaseModel):
    slippage_bps_per_second: float = Field(ge=0)
    max_slippage_bps: float = Field(ge=0)


class StatsSettings(BaseModel):
    bootstrap_samples: int = Field(ge=100)
    ci_level: float = Field(gt=0, lt=1)
    alpha: float = Field(gt=0, lt=1)
    dsr_pass_threshold: float = Field(gt=0, lt=1)
    bradley_terry_prior: float = Field(ge=0)
    bradley_terry_max_iter: int = Field(ge=1)
    bradley_terry_tol: float = Field(gt=0)
    seed: int


class DiagnosticsSettings(BaseModel):
    flat_epsilon: float = Field(ge=0)
    trade_epsilon: float = Field(ge=0)


class ExperimentSettings(BaseModel):
    """Parameters of the report experiments (``make report``); see docs/report.md."""

    reports_subdir: str
    seed: int
    replications: int = Field(ge=10)
    bootstrap_samples: int = Field(ge=100)
    periods_per_year: float = Field(gt=0)
    steps_per_run: int = Field(ge=3)
    episodes: int = Field(ge=1)
    seeds: int = Field(ge=1)
    variants: int = Field(ge=1)
    skill_sharpes: list[float] = Field(min_length=2)
    factor_betas: list[float] = Field(min_length=2)
    factor_vol: float = Field(ge=0)
    replicate_offset_vol: float = Field(ge=0)
    win_probability_threshold: float = Field(gt=0.5, lt=1)
    designs: list[tuple[int, int, int]] = Field(min_length=1)
    token_multipliers: list[float] = Field(min_length=1)
    latency_rates_bps_per_s: list[float] = Field(min_length=1)


class ArenaSettings(BaseSettings):
    """All Agent Arena configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ARENA_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data_root: Path = Field(validation_alias=AliasChoices("TRADEQ_DATA_DIR", "ARENA_DATA_ROOT"))
    data_subdir: str
    seasons_dir: Path
    prompts_file: Path
    max_concurrent_runs: int = Field(ge=1)
    log_level: str
    service: ServiceSettings
    gateway: GatewaySettings
    ledger: LedgerSettings
    leakage: LeakageSettings
    providers: dict[str, ProviderSettings]
    pricing: dict[str, ModelPrice]
    accounting: AccountingSettings
    fees: FeeSettings
    latency: LatencySettings
    stats: StatsSettings
    diagnostics: DiagnosticsSettings
    experiments: ExperimentSettings

    @property
    def arena_dir(self) -> Path:
        """This project's data subtree, ``$TRADEQ_DATA_DIR/agent-arena``."""
        return self.data_root / self.data_subdir

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = Path(os.environ.get(SETTINGS_FILE_ENV, str(DEFAULT_SETTINGS_FILE)))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
        )


def load_settings() -> ArenaSettings:
    """Load settings from the YAML file named by ``ARENA_SETTINGS_FILE`` plus the environment."""
    return ArenaSettings()
