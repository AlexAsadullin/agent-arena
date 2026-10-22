from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_arena.accounting import Ledger, dec, latency_cost, latency_slippage_bps, token_cost
from agent_arena.agents.base import Usage
from agent_arena.settings import LatencySettings, ModelPrice

PRICES = {"m": ModelPrice(input=4.0, output=20.0)}


def test_token_cost_uses_price_table() -> None:
    usage = Usage(model="m", tokens_in=1_000_000, tokens_out=500_000, latency_ms=0)
    assert token_cost(usage, PRICES) == Decimal("14")
    with pytest.raises(KeyError):
        token_cost(Usage(model="x", tokens_in=1, tokens_out=1, latency_ms=0), PRICES)


def test_latency_slippage_is_linear_then_capped() -> None:
    settings = LatencySettings(slippage_bps_per_second=2.0, max_slippage_bps=5.0)
    assert latency_slippage_bps(1500, settings) == pytest.approx(3.0)
    assert latency_slippage_bps(10_000, settings) == 5.0
    assert latency_cost(Decimal(10_000), 1500, settings) == Decimal("3")


def test_ledger_equity_and_weights() -> None:
    ledger = Ledger(1000.0)
    ledger.apply_fill("A", 2.0, 100.0, 0.1)
    assert ledger.cash == Decimal("799.9")
    assert ledger.equity({"A": 110.0}) == Decimal("1019.9")
    ledger.charge_tokens(Decimal("0.4"))
    ledger.charge_latency(Decimal("0.5"))
    assert ledger.token_costs == Decimal("0.4")
    assert ledger.equity({"A": 100.0}) == Decimal("999.0")
    assert ledger.weights({"A": 100.0})["A"] == pytest.approx(200 / 999)


def test_bankrupt_ledger_has_zero_weights() -> None:
    ledger = Ledger(10.0)
    ledger.apply_fill("A", 1.0, 100.0, 0.0)
    assert ledger.weights({"A": 1.0}) == {"A": 0.0}


@given(
    st.floats(1, 1e6, allow_nan=False),
    st.floats(0.001, 100, allow_nan=False),
    st.floats(0.01, 1e5, allow_nan=False),
    st.floats(0, 10, allow_nan=False),
)
def test_round_trip_costs_only_fees(capital: float, qty: float, price: float, fee: float) -> None:
    ledger = Ledger(capital)
    ledger.apply_fill("A", qty, price, fee)
    ledger.apply_fill("A", -qty, price, fee)
    assert ledger.position("A") == 0.0
    # Decimal keeps 28 significant digits, so the residual is far below a cent of the notional.
    expected = dec(capital) - 2 * dec(fee)
    assert abs(ledger.equity({"A": price}) - expected) <= Decimal("1e-20") * (1 + dec(qty * price))
