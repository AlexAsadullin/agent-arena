"""Wire models for the I9 gateway (contract 6.6).

Contract 6.6 fixes tool names and arguments; the result shapes below follow the I9 gateway's
published output schemas (risk-gated-mcp-gateway README, "Contracts"). The models ignore fields
the arena does not read, so a richer response still parses.

Refusals arrive as MCP tool errors (``isError: true``) with the body
``{refused: true, code, message, details, session_status}``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_arena.timeutil import parse_iso, to_ns

SCHEMA_VERSION = "1"


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = SCHEMA_VERSION


class Refusal(_Wire):
    """A structured refusal (budget breach, time-travel attempt, pre-trade reject, frozen session)."""

    code: str
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    session_status: str | None = None

    @property
    def frozen(self) -> bool:
        return self.session_status == "frozen"


class SessionBudgets(BaseModel):
    """Budget overrides sent at session creation (I9 ``BudgetOverrides``; unset fields use I9 defaults).

    ``capital`` is the maximum gross exposure after a trade (quote currency), not the account size.
    """

    capital: float | None = None
    max_drawdown: float | None = None
    max_order_notional: float | None = None
    max_orders_per_window: int | None = None
    order_window_s: float | None = None
    max_tokens: int | None = None
    max_token_cost_usd: float | None = None
    max_latency_ms: float | None = None
    max_total_latency_ms: float | None = None


class CreateSessionRequest(_Wire):
    agent_id: str
    mode: Literal["research", "paper", "live"]
    known_at: str
    budgets: SessionBudgets


class SessionInfo(_Wire):
    session_id: str
    agent_id: str | None = None
    known_at: str | None = None
    status: str | None = None
    api_key: str | None = None
    mcp_path: str | None = None


class ClockRequest(_Wire):
    known_at: str


class ClockInfo(_Wire):
    known_at: str | None = None
    status: str | None = None


def _ts_ns(value: int | str) -> int:
    """Row timestamp as int64 ns: I1 stores ints, I2 JSON answers may carry ISO-8601 strings."""
    return value if isinstance(value, int) else to_ns(parse_iso(value))


class TradeRow(BaseModel):
    """One row of the I1 ``trades`` channel as I9 returns it from I2.

    I2 rows carry ``valid_time`` (when the trade happened) and ``knowledge_time``; raw I1 rows carry
    ``ts_exchange`` and ``ts_local``. The trade time is the first one present, in that order.
    """

    model_config = ConfigDict(extra="ignore")
    ts_exchange: int | str | None = None
    valid_time: int | str | None = None
    ts_local: int | str | None = None
    price: float
    size: float
    side: str | None = None

    @property
    def ts(self) -> int:
        for value in (self.ts_exchange, self.valid_time, self.ts_local):
            if value is not None:
                return _ts_ns(value)
        raise ValueError("trade row without timestamp")


class MarketData(_Wire):
    venue: str | None = None
    symbol: str | None = None
    channel: str | None = None
    known_at: str | None = None
    rows: list[TradeRow] = Field(default_factory=list)


class FeatureRow(BaseModel):
    """Long-format feature row; other row shapes are ignored by the arena."""

    model_config = ConfigDict(extra="ignore")
    symbol: str
    name: str
    value: float | None


class Features(_Wire):
    """Feature rows from I2. I2 answers in wide format: one row per symbol and ``valid_time``, one column
    per requested feature. Long rows (``symbol, name, value``) are accepted too."""

    rows: list[dict[str, Any]] = Field(default_factory=list)

    def values(self, names: list[str]) -> list[FeatureRow]:
        """Latest value of each requested feature per symbol."""
        latest: dict[tuple[str, str], tuple[int, float | None]] = {}
        for i, row in enumerate(self.rows):
            symbol = row.get("symbol")
            if not isinstance(symbol, str):
                continue
            stamp = row.get("valid_time")
            order = _ts_ns(stamp) if isinstance(stamp, int | str) else i
            if {"name", "value"} <= row.keys():
                cells = [(str(row["name"]), row["value"])]
            else:
                cells = [(n, row[n]) for n in names if n in row]
            for name, value in cells:
                number = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
                if (symbol, name) not in latest or order >= latest[(symbol, name)][0]:
                    latest[(symbol, name)] = (order, number)
        return [FeatureRow(symbol=s, name=n, value=v) for (s, n), (_, v) in sorted(latest.items())]


class ForecastItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    horizon: int
    mean: float
    quantiles: dict[str, float] = Field(default_factory=dict)


class Forecast(_Wire):
    model: str | None = None
    forecasts: list[ForecastItem] = Field(default_factory=list)
    model_cutoff: str | None = None


class Order(BaseModel):
    """Order per contract 6.4. ``filled_qty``, ``avg_price`` and ``fee`` are optional fill details."""

    model_config = ConfigDict(extra="ignore")
    order_id: str
    status: Literal["accepted", "rejected", "partially_filled", "filled", "cancelled"]
    reject_reason: str | None = None
    qty: float | None = None
    price: float | None = None
    filled_qty: float | None = None
    avg_price: float | None = None
    fee: float | None = None


class OrderResult(_Wire):
    order: Order
    reference_price: float | None = None
    notional: float | None = None


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    qty: float
    avg_price: float | None = None
    mark_price: float | None = None


class Positions(_Wire):
    positions: list[Position] = Field(default_factory=list)


class AccountInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    account_id: str
    equity: float | None = None
    balance: float | None = None


class Account(_Wire):
    account: AccountInfo
    peak_equity: float | None = None
    drawdown: float | None = None


class BudgetStatus(_Wire):
    status: str
    freeze: dict[str, Any] | None = None
    limits: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    remaining: dict[str, float] = Field(default_factory=dict)

    @property
    def frozen(self) -> bool:
        return self.status == "frozen"

    @property
    def frozen_reason(self) -> str | None:
        if self.freeze is None:
            return None
        return str(self.freeze.get("reason") or self.freeze.get("kind") or "frozen")


class UsageAck(_Wire):
    model: str | None = None
    cost_usd: float | None = None
    remaining: dict[str, float] = Field(default_factory=dict)


class BacktestResult(_Wire):
    run_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class VerdictResult(_Wire):
    strategy_id: str
    verdict: dict[str, Any] = Field(default_factory=dict)
    leakage_status: str | None = None
    promotable: bool | None = None
    reasons: list[str] = Field(default_factory=list)
