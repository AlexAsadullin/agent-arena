"""Net-of-intelligence-cost accounting.

The account currency is the token price currency (USD). Balances, fees and charges are kept as
``decimal.Decimal`` so that small token charges are never lost to float rounding against a large
cash balance; returns handed to statistics are float64.

Intelligence cost per decision has two parts:

* token cost = ``tokens_in * price_in + tokens_out * price_out`` (USD per million tokens, from
  ``settings.pricing``);
* latency cost = traded notional * ``min(max_slippage_bps, slippage_bps_per_second * latency_s)``
  / 10^4. The arena freezes the market clock while the agent thinks, so without this charge a slow
  agent would trade at prices that a live agent could no longer get.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from agent_arena.agents.base import Usage
from agent_arena.settings import LatencySettings, ModelPrice

TOKENS_PER_MILLION = Decimal(1_000_000)
BPS = Decimal(10_000)
MS_PER_SECOND = 1000.0


def dec(value: float) -> Decimal:
    """Exact decimal of the float's shortest repr (avoids binary-expansion noise)."""
    return Decimal(repr(float(value)))


def token_cost(usage: Usage, pricing: Mapping[str, ModelPrice]) -> Decimal:
    price = pricing.get(usage.model)
    if price is None:
        raise KeyError(f"no price for model {usage.model!r}")
    return (usage.tokens_in * dec(price.input) + usage.tokens_out * dec(price.output)) / TOKENS_PER_MILLION


def latency_slippage_bps(latency_ms: float, settings: LatencySettings) -> float:
    return min(settings.max_slippage_bps, settings.slippage_bps_per_second * latency_ms / MS_PER_SECOND)


def latency_cost(traded_notional: Decimal, latency_ms: float, settings: LatencySettings) -> Decimal:
    return traded_notional * dec(latency_slippage_bps(latency_ms, settings)) / BPS


class Ledger:
    """Cash and positions of one run, mirrored from gateway fills, plus intelligence charges."""

    def __init__(self, capital: float) -> None:
        self.cash = dec(capital)
        self.positions: dict[str, Decimal] = {}
        self.fees = Decimal(0)
        self.token_costs = Decimal(0)
        self.latency_costs = Decimal(0)

    def apply_fill(self, symbol: str, signed_qty: float, price: float, fee: float) -> None:
        qty = dec(signed_qty)
        self.positions[symbol] = self.positions.get(symbol, Decimal(0)) + qty
        self.cash -= qty * dec(price) + dec(fee)
        self.fees += dec(fee)

    def charge_tokens(self, amount: Decimal) -> None:
        self.cash -= amount
        self.token_costs += amount

    def charge_latency(self, amount: Decimal) -> None:
        self.cash -= amount
        self.latency_costs += amount

    def position(self, symbol: str) -> float:
        return float(self.positions.get(symbol, Decimal(0)))

    def equity(self, prices: Mapping[str, float]) -> Decimal:
        return self.cash + sum((qty * dec(prices[s]) for s, qty in self.positions.items() if qty != 0), Decimal(0))

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        equity = float(self.equity(prices))
        if equity <= 0:
            return dict.fromkeys(prices, 0.0)
        return {s: self.position(s) * prices[s] / equity for s in prices}
