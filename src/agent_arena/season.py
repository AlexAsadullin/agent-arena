"""Season definitions: who competes, on what, for how long, across which seeds and prompt variants.

A season is a grid of *cells* ``(episode, seed)``. Every agent sees the same market path in a
given cell, so comparisons between agents are paired. LLM agents are additionally run once per
prompt variant inside each cell; baselines ignore prompts and run once per cell.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class RegimeConfig(BaseModel):
    """Synthetic return process for one episode, used only by the fake gateway.

    Per-bar log return: ``r_t = drift + phi * r_{t-1} + vol * eps_t``.
    """

    name: str
    drift: float
    vol: float = Field(gt=0)
    phi: float = Field(gt=-1, lt=1)


class MarketConfig(BaseModel):
    """Synthetic market served by the fake gateway."""

    seed: int
    initial_prices: dict[str, float]
    trades_per_bar: int = Field(ge=1)
    regimes: list[RegimeConfig] = Field(min_length=1)
    fee_bps: float = Field(ge=0)


class BudgetConfig(BaseModel):
    """Per-session budgets requested from the gateway when a session is created."""

    capital: float = Field(gt=0)
    max_drawdown: float = Field(gt=0, le=1)
    max_order_notional: float = Field(gt=0)
    max_position_notional: float = Field(gt=0)
    max_orders_per_step: int = Field(ge=1)
    token_spend_usd: float = Field(ge=0)
    max_latency_ms: float = Field(gt=0)


class LLMAgentConfig(BaseModel):
    """An LLM agent: a provider adapter, a model id and the prompt variants it runs under."""

    kind: Literal["llm"] = "llm"
    id: str
    provider: str
    model: str
    # Parameters for the scripted double (ignored by real providers).
    behaviour: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class BaselineAgentConfig(BaseModel):
    """A quant baseline competing under the same rules."""

    kind: Literal["baseline"] = "baseline"
    id: str
    strategy: Literal["buy_and_hold", "ts_momentum", "mean_reversion", "random"]
    params: dict[str, float] = Field(default_factory=dict)


AgentConfig = Annotated[LLMAgentConfig | BaselineAgentConfig, Field(discriminator="kind")]


class SeasonConfig(BaseModel):
    season_id: str
    description: str = ""
    venue: str
    symbols: list[str] = Field(min_length=1)
    channel: str
    bar_seconds: int = Field(gt=0)
    start: datetime
    steps_per_episode: int = Field(ge=2)
    lookback_bars: int = Field(ge=2)
    episode_gap_bars: int = Field(ge=0)
    episodes: int = Field(ge=1)
    seeds: list[int] = Field(min_length=1)
    prompt_variants: list[str] = Field(min_length=1)
    max_weight: float = Field(gt=0)
    periods_per_year: float = Field(gt=0)
    features: list[str] = Field(default_factory=list)
    forecast_model: str | None = None
    forecast_horizon: int = Field(default=1, ge=1)
    budgets: BudgetConfig
    market: MarketConfig
    agents: list[AgentConfig] = Field(min_length=1)

    @field_validator("start")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("season start must be timezone-aware (UTC)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check(self) -> SeasonConfig:
        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        missing = set(self.symbols) - set(self.market.initial_prices)
        if missing:
            raise ValueError(f"market.initial_prices lacks symbols: {sorted(missing)}")
        return self

    @property
    def bar(self) -> timedelta:
        return timedelta(seconds=self.bar_seconds)

    @property
    def episode_span_bars(self) -> int:
        """Bars an episode occupies: decision steps plus the final mark."""
        return self.steps_per_episode + 1

    def episode_start(self, episode: int) -> datetime:
        """First decision time of ``episode``; history before it serves as look-back."""
        offset = self.lookback_bars + episode * (self.episode_span_bars + self.episode_gap_bars)
        return self.start + offset * self.bar

    def step_time(self, episode: int, step: int) -> datetime:
        return self.episode_start(episode) + step * self.bar

    @property
    def end(self) -> datetime:
        """Last timestamp the season touches (the final mark of the last episode)."""
        return self.step_time(self.episodes - 1, self.steps_per_episode)

    @property
    def total_bars(self) -> int:
        return int((self.end - self.start) / self.bar) + 1

    def regime(self, episode: int) -> RegimeConfig:
        return self.market.regimes[episode % len(self.market.regimes)]

    @property
    def llm_agents(self) -> list[LLMAgentConfig]:
        return [a for a in self.agents if isinstance(a, LLMAgentConfig)]

    @property
    def baseline_agents(self) -> list[BaselineAgentConfig]:
        return [a for a in self.agents if isinstance(a, BaselineAgentConfig)]


def load_season(path: Path) -> SeasonConfig:
    """Read and validate a season YAML file."""
    with path.open(encoding="utf-8") as fh:
        return SeasonConfig.model_validate(yaml.safe_load(fh))
