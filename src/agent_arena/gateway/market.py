"""Synthetic market for one season episode, served by the fake gateway.

Bar closes follow an AR(1) log-return process per symbol (see :class:`RegimeConfig`). Each bar
also carries ``trades_per_bar`` synthetic trades in the I1 ``trades`` schema; the last trade of a
bar prints exactly at the bar close and is stamped at the bar boundary, so a client that takes the
last trade at or before a boundary recovers the close.
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from agent_arena.season import SeasonConfig
from agent_arena.timeutil import NS_PER_SECOND, to_ns

# Intra-bar trade prices wander around the close by a fraction of the bar volatility.
_INTRABAR_NOISE_FRACTION = 0.25


def cell_seed(*parts: int | str) -> int:
    """Stable 63-bit seed derived from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


@dataclass(frozen=True)
class Trade:
    ts_exchange: int
    price: float
    size: float
    side: str
    trade_id: str


class SyntheticMarket:
    """Deterministic bar and trade history for one episode (identical for every agent and seed)."""

    def __init__(self, season: SeasonConfig, episode: int) -> None:
        self.season = season
        self.episode = episode
        self.first_bar = season.episode_start(episode) - season.lookback_bars * season.bar
        self.n_bars = season.lookback_bars + season.episode_span_bars
        self.bar_ns = season.bar_seconds * NS_PER_SECOND
        self.first_bar_ns = to_ns(self.first_bar)
        regime = season.regime(episode)
        self.closes: dict[str, NDArray[np.float64]] = {}
        self._trades: dict[str, list[Trade]] = {}
        self._trade_ts: dict[str, list[int]] = {}
        for symbol in season.symbols:
            rng = np.random.default_rng(cell_seed(season.market.seed, episode, symbol))
            eps = rng.standard_normal(self.n_bars)
            rets = np.empty(self.n_bars)
            prev = 0.0
            for i in range(self.n_bars):
                prev = regime.drift + regime.phi * prev + regime.vol * eps[i]
                rets[i] = prev
            rets[0] = 0.0
            closes = season.market.initial_prices[symbol] * np.exp(np.cumsum(rets))
            self.closes[symbol] = closes
            self._trades[symbol] = self._make_trades(symbol, closes, regime.vol, rng)
            self._trade_ts[symbol] = [t.ts_exchange for t in self._trades[symbol]]

    def _make_trades(
        self,
        symbol: str,
        closes: NDArray[np.float64],
        vol: float,
        rng: np.random.Generator,
    ) -> list[Trade]:
        n = self.season.market.trades_per_bar
        trades: list[Trade] = []
        for i, close in enumerate(closes):
            bar_end = self.first_bar_ns + i * self.bar_ns
            noise = rng.standard_normal(n) * vol * _INTRABAR_NOISE_FRACTION
            noise[-1] = 0.0
            sizes = rng.exponential(1.0, n)
            sides = rng.integers(0, 2, n)
            for j in range(n):
                ts = bar_end - (n - 1 - j) * (self.bar_ns // n)
                trades.append(
                    Trade(
                        ts_exchange=ts,
                        price=float(close * np.exp(noise[j])),
                        size=float(sizes[j]),
                        side="buy" if sides[j] else "sell",
                        trade_id=f"{symbol}-{i}-{j}",
                    )
                )
        return trades

    def bar_index(self, ts: datetime) -> int:
        """Index of the last bar whose close is at or before ``ts`` (-1 if none)."""
        idx = (to_ns(ts) - self.first_bar_ns) // self.bar_ns
        return int(min(idx, self.n_bars - 1))

    def price_at(self, symbol: str, ts: datetime) -> float:
        """Last close at or before ``ts``."""
        idx = self.bar_index(ts)
        if idx < 0:
            raise ValueError(f"no data for {symbol} at {ts.isoformat()}")
        return float(self.closes[symbol][idx])

    def closes_until(self, symbol: str, ts: datetime) -> NDArray[np.float64]:
        """All closes at or before ``ts``."""
        return self.closes[symbol][: self.bar_index(ts) + 1]

    def trades(self, symbol: str, start: datetime, end: datetime) -> list[Trade]:
        stamps = self._trade_ts[symbol]
        lo = bisect.bisect_left(stamps, to_ns(start))
        hi = bisect.bisect_right(stamps, to_ns(end))
        return self._trades[symbol][lo:hi]
