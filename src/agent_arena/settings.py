"""Typed settings: defaults from ``config/settings.yaml``, overridden by ``ARENA_*`` env variables.

Precedence (highest first): init kwargs, environment, ``.env``, the YAML file. Secrets (API keys,
the gateway admin key) are only ever read from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

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
    min_order_notional: float = Field(ge=0)


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
    latency: LatencySettings
    stats: StatsSettings
    diagnostics: DiagnosticsSettings

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
