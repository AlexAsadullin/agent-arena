"""Deterministic scripted LLM double.

It implements :class:`LLMProvider`, reads the rendered prompt exactly as a real model would get
it, extracts the observation JSON between ``<observation>`` tags, and answers with text. Its
behaviour, token counts and latency are fully determined by the season config, the run seed and
the prompt text, so tests and the offline demo are reproducible while still exercising parsing,
refusal detection, cost accounting and latency charging.

Behaviours (``behaviour`` in the season config):

* ``trend_chaser`` - volatility-scaled momentum with large, noisy sizes: an over-trader.
* ``cautious_contrarian`` - small mean-reversion positions with a no-trade band.
* ``martingale_gambler`` - multiplies its size after every losing step, sometimes answers with
  malformed text or refuses.

Prompt variants move the double through ``keyword_scales``: if a keyword appears in the prompt,
the position size is multiplied by its scale (e.g. "aggressive" -> 1.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from agent_arena.agents.providers import Completion, ProviderError
from agent_arena.gateway.market import cell_seed

_OBSERVATION_RE = re.compile(r"<observation>\s*(\{.*?\})\s*</observation>", re.DOTALL)


class ScriptedProvider:
    def __init__(self, model: str, behaviour: str, params: dict[str, Any], seed: int) -> None:
        self.model = model
        self.behaviour = behaviour
        self.params = params
        self.seed = seed
        self._rng: np.random.Generator | None = None
        self._last_equity: float | None = None
        self._size = 0.0
        self._direction: dict[str, float] = {}

    def _p(self, name: str) -> Any:
        try:
            return self.params[name]
        except KeyError as exc:
            raise ProviderError(f"scripted {self.behaviour} needs param {name!r}") from exc

    async def complete(self, system: str, user: str) -> Completion:
        if self._rng is None:
            # Seeded by the prompt text too, so two prompt variants are two different "samples".
            prompt_hash = hashlib.sha256((system + "\x00" + user.split("<observation>")[0]).encode()).hexdigest()
            self._rng = np.random.default_rng(cell_seed(self.model, self.seed, prompt_hash))
        match = _OBSERVATION_RE.search(user)
        if match is None:
            raise ProviderError("scripted double: prompt has no <observation> block")
        obs = json.loads(match.group(1))
        scale = self._prompt_scale(system + "\n" + user)
        # Refusals are signalled in text only, as many real models do; the agent's parser must spot them.
        text = self._respond(obs, scale, self._rng)
        chars_per_token = float(self._p("chars_per_token"))
        tokens_in = math.ceil((len(system) + len(user)) / chars_per_token)
        tokens_out = math.ceil(len(text) / chars_per_token)
        latency_ms = (
            float(self._p("base_latency_ms"))
            + float(self._p("ms_per_output_token")) * tokens_out
            + abs(float(self._rng.normal(0.0, float(self._p("latency_jitter_ms")))))
        )
        return Completion(
            text=text,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

    def _prompt_scale(self, prompt: str) -> float:
        lowered = prompt.lower()
        scale = 1.0
        for keyword, factor in dict(self.params.get("keyword_scales", {})).items():
            if str(keyword).lower() in lowered:
                scale *= float(factor)
        return scale

    def _respond(self, obs: dict[str, Any], scale: float, rng: np.random.Generator) -> str:
        if self.behaviour == "trend_chaser":
            targets = self._trend(obs, scale, rng)
        elif self.behaviour == "cautious_contrarian":
            targets = self._contrarian(obs, scale)
        elif self.behaviour == "martingale_gambler":
            roll = float(rng.uniform())
            if roll < float(self._p("refusal_rate")):
                return str(self._p("refusal_text"))
            if roll < float(self._p("refusal_rate")) + float(self._p("error_rate")):
                return "I think the market looks bullish, buy everything!"
            targets = self._martingale(obs, scale)
        else:
            raise ProviderError(f"unknown scripted behaviour {self.behaviour}")
        words = int(self._p("rationale_words"))
        rationale = " ".join(["because"] * words)
        return json.dumps({"targets": targets, "rationale": rationale})

    @staticmethod
    def _log_ret(closes: list[float], lookback: int) -> float:
        if len(closes) <= lookback:
            return 0.0
        return math.log(closes[-1] / closes[-1 - lookback])

    def _trend(self, obs: dict[str, Any], scale: float, rng: np.random.Generator) -> dict[str, float]:
        lookback = int(self._p("lookback"))
        bound = float(obs["max_weight"])
        out: dict[str, float] = {}
        for sym in obs["symbols"]:
            closes = obs["closes"][sym]
            rets = np.diff(np.log(np.asarray(closes)))
            vol = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
            signal = self._log_ret(closes, lookback) / (vol * math.sqrt(lookback)) if vol > 0 else 0.0
            raw = scale * float(self._p("gain")) * signal + float(self._p("noise")) * float(rng.normal())
            out[sym] = round(max(-bound, min(bound, raw)), 4)
        return out

    def _contrarian(self, obs: dict[str, Any], scale: float) -> dict[str, float]:
        lookback = int(self._p("lookback"))
        band = float(self._p("rebalance_band"))
        out: dict[str, float] = {}
        for sym in obs["symbols"]:
            window = np.asarray(obs["closes"][sym][-lookback:])
            sd = float(window.std(ddof=1)) if window.size > 1 else 0.0
            z = float((window[-1] - window.mean()) / sd) if sd > 0 else 0.0
            target = -scale * float(self._p("weight")) * max(-1.0, min(1.0, z / float(self._p("z_scale"))))
            current = float(obs["weights"].get(sym, 0.0))
            out[sym] = round(current if abs(target - current) < band else target, 4)
        return out

    def _martingale(self, obs: dict[str, Any], scale: float) -> dict[str, float]:
        base = float(self._p("base_weight")) * scale
        equity = float(obs["equity"])
        if self._last_equity is not None and equity < self._last_equity:
            self._size = min(self._size * float(self._p("multiplier")), float(obs["max_weight"]) * 2)
        else:
            self._size = base
        self._last_equity = equity
        out: dict[str, float] = {}
        for sym in obs["symbols"]:
            closes = obs["closes"][sym]
            last = self._log_ret(closes, 1)
            direction = self._direction.get(sym, 1.0) if last == 0 else math.copysign(1.0, last)
            self._direction[sym] = direction
            out[sym] = round(direction * self._size, 4)
        return out
