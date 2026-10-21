"""Wire models for the I9 gateway (contract 6.6).

Contract 6.6 fixes tool names and arguments but not result shapes. These models state what the
arena reads from each result. They ignore unknown fields, so a richer gateway response still
parses, and every field the arena can live without is optional.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = SCHEMA_VERSION


class Refusal(_Wire):
    """A structured refusal from the gateway (budget breach, time-travel attempt, frozen session)."""

    code: str
    reason: str
    frozen: bool = False


class SessionBudgets(_Wire):
    capital: float
    max_drawdown: float
    max_order_notional: float
    max_position_notional: float
    max_orders_per_step: int
    token_spend_usd: float
    max_latency_ms: float


class CreateSessionRequest(_Wire):
    agent_id: str
    mode: Literal["research", "paper", "live"]
    known_at: str
    budgets: SessionBudgets


class SessionInfo(_Wire):
    session_id: str
    agent_id: str | None = None
    known_at: str | None = None
    api_key: str | None = None


class ClockRequest(_Wire):
    known_at: str


class ClockInfo(_Wire):
    known_at: str


class TradeRow(BaseModel):
    """One row of the I1 ``trades`` channel."""

    model_config = ConfigDict(extra="ignore")
    ts_exchange: int | None = None
    ts_local: int | None = None
    price: float
    size: float
    side: str | None = None

    @property
    def ts(self) -> int:
        value = self.ts_exchange if self.ts_exchange is not None else self.ts_local
        if value is None:
            raise ValueError("trade row without timestamp")
        return value


class MarketData(_Wire):
    venue: str | None = None
    symbol: str | None = None
    channel: str | None = None
    rows: list[TradeRow] = Field(default_factory=list)


class FeatureValue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    name: str
    value: float | None


class Features(_Wire):
    features: list[FeatureValue] = Field(default_factory=list)


class ForecastItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    horizon: int
    mean: float
    quantiles: dict[str, float] = Field(default_factory=dict)


class Forecast(_Wire):
    forecasts: list[ForecastItem] = Field(default_factory=list)
    model_cutoff: str | None = None


class OrderResult(_Wire):
    """Order per contract 6.4, plus optional fill details reported by the gateway."""

    order_id: str | None = None
    status: Literal["accepted", "rejected", "partially_filled", "filled", "cancelled"]
    reject_reason: str | None = None
    filled_qty: float | None = None
    avg_price: float | None = None
    fee: float | None = None


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    qty: float
    avg_price: float | None = None


class Positions(_Wire):
    positions: list[Position] = Field(default_factory=list)


class Account(_Wire):
    account_id: str | None = None
    equity: float
    cash: float


class BudgetStatus(_Wire):
    limits: dict[str, float] = Field(default_factory=dict)
    used: dict[str, float] = Field(default_factory=dict)
    frozen: bool = False
    frozen_reason: str | None = None


class UsageAck(_Wire):
    accepted: bool = True
    token_spend_usd: float | None = None
    frozen: bool = False


class BacktestResult(_Wire):
    run_id: str
    trial_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class VerdictResult(_Wire):
    strategy_id: str
    passed: bool
    dsr: float | None = None
    reasons: list[str] = Field(default_factory=list)
