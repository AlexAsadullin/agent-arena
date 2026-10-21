"""Quant baselines: buy-and-hold, time-series momentum, mean reversion and random.

All parameters come from the season config. Each baseline returns target weights clipped to the
season's ``max_weight``; none of them use tokens, so their intelligence cost is zero.
"""

from __future__ import annotations

import math

import numpy as np

from agent_arena.agents.base import Decision, Observation
from agent_arena.gateway.market import cell_seed
from agent_arena.season import BaselineAgentConfig


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _log_return(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback:
        return None
    return math.log(closes[-1] / closes[-1 - lookback])


class BaselineAgent:
    """Rule-based competitor. ``params`` keys depend on the strategy (see each method)."""

    def __init__(self, config: BaselineAgentConfig, seed: int) -> None:
        self.config = config
        self._rng = np.random.default_rng(cell_seed(config.id, seed))

    @property
    def agent_id(self) -> str:
        return self.config.id

    @property
    def is_llm(self) -> bool:
        return False

    def _param(self, name: str) -> float:
        try:
            return self.config.params[name]
        except KeyError as exc:
            raise ValueError(f"baseline {self.config.id} needs param {name!r}") from exc

    async def decide(self, obs: Observation) -> Decision:
        strategy = self.config.strategy
        targets = {s: _clip(self._target(strategy, obs, s), obs.max_weight) for s in obs.symbols}
        return Decision(targets=targets, rationale=strategy)

    def _target(self, strategy: str, obs: Observation, symbol: str) -> float:
        closes = obs.closes[symbol]
        weight = self._param("weight")
        if strategy == "buy_and_hold":
            return weight
        if strategy == "ts_momentum":
            # Moskowitz, Ooi & Pedersen (2012): position = sign of trailing return.
            ret = _log_return(closes, int(self._param("lookback")))
            return 0.0 if ret is None else weight * float(np.sign(ret))
        if strategy == "mean_reversion":
            return weight * self._reversion_signal(closes)
        if strategy == "random":
            return float(self._rng.uniform(-weight, weight))
        raise ValueError(f"unknown baseline strategy {strategy}")

    def _reversion_signal(self, closes: list[float]) -> float:
        """Negative z-score of the last price versus its trailing mean, scaled into [-1, 1]."""
        lookback = int(self._param("lookback"))
        if len(closes) < lookback:
            return 0.0
        window = np.asarray(closes[-lookback:])
        sd = float(window.std(ddof=1))
        if sd == 0.0:
            return 0.0
        z = (window[-1] - window.mean()) / sd
        return _clip(-float(z) / self._param("z_scale"), 1.0)
