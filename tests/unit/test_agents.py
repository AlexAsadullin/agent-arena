from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent_arena.agents.base import Observation
from agent_arena.agents.baselines import BaselineAgent
from agent_arena.agents.llm import PromptVariant, render_prompt
from agent_arena.agents.providers import ProviderError
from agent_arena.agents.scripted import ScriptedProvider
from agent_arena.season import BaselineAgentConfig

COMMON = {"chars_per_token": 4, "base_latency_ms": 100, "ms_per_output_token": 1, "latency_jitter_ms": 10, "rationale_words": 3}
VARIANT = PromptVariant(system="be $symbols", user="<observation>\n$observation_json\n</observation>")


def obs(closes: list[float], weights: float = 0.0, equity: float = 1000.0) -> Observation:
    return Observation(
        season_id="s",
        episode=0,
        step=0,
        seed=0,
        known_at=datetime(2025, 1, 1, tzinfo=UTC),
        symbols=["A"],
        closes={"A": closes},
        weights={"A": weights},
        equity=equity,
        cash=equity,
        max_weight=1.0,
    )


UP = [100.0 * 1.01**i for i in range(30)]


async def ask(provider: ScriptedProvider, o: Observation, variant: PromptVariant = VARIANT) -> dict[str, float]:
    system, user = render_prompt(variant, o)
    out = await provider.complete(system, user)
    assert out.tokens_in > 0
    assert out.latency_ms >= 100
    return dict(json.loads(out.text)["targets"])


async def test_trend_chaser_is_deterministic_and_follows_trend() -> None:
    params = {**COMMON, "lookback": 5, "gain": 0.5, "noise": 0.0, "keyword_scales": {"aggressive": 2.0}}
    a = await ask(ScriptedProvider("m", "trend_chaser", params, seed=1), obs(UP))
    b = await ask(ScriptedProvider("m", "trend_chaser", params, seed=1), obs(UP))
    assert a == b
    assert a["A"] > 0
    loud = PromptVariant(system="be aggressive", user=VARIANT.user)
    c = await ask(ScriptedProvider("m", "trend_chaser", params, seed=1), obs(UP), loud)
    assert c["A"] >= a["A"]


async def test_contrarian_band_holds_position() -> None:
    params = {**COMMON, "lookback": 10, "z_scale": 2.0, "weight": 0.3, "rebalance_band": 0.5}
    held = await ask(ScriptedProvider("m", "cautious_contrarian", params, 0), obs(UP, weights=0.1))
    assert held["A"] == pytest.approx(0.1)
    params["rebalance_band"] = 0.0
    moved = await ask(ScriptedProvider("m", "cautious_contrarian", params, 0), obs(UP, weights=0.1))
    assert moved["A"] < 0


async def test_martingale_doubles_after_losses() -> None:
    params = {**COMMON, "base_weight": 0.1, "multiplier": 2.0, "refusal_rate": 0.0, "error_rate": 0.0, "refusal_text": "no"}
    p = ScriptedProvider("m", "martingale_gambler", params, 0)
    first = await ask(p, obs(UP, equity=1000))
    second = await ask(p, obs(UP, equity=990))
    third = await ask(p, obs(UP, equity=995))
    assert abs(second["A"]) == pytest.approx(2 * abs(first["A"]))
    assert abs(third["A"]) == pytest.approx(abs(first["A"]))


async def test_martingale_refuses_and_errors() -> None:
    params = {**COMMON, "base_weight": 0.1, "multiplier": 2.0, "refusal_rate": 1.0, "error_rate": 0.0, "refusal_text": "I cannot"}
    system, user = render_prompt(VARIANT, obs(UP))
    assert (await ScriptedProvider("m", "martingale_gambler", params, 0).complete(system, user)).text == "I cannot"
    params.update(refusal_rate=0.0, error_rate=1.0)
    text = (await ScriptedProvider("m", "martingale_gambler", params, 0).complete(system, user)).text
    assert "{" not in text


async def test_scripted_failure_modes() -> None:
    with pytest.raises(ProviderError, match="observation"):
        await ScriptedProvider("m", "trend_chaser", COMMON, 0).complete("s", "no tags")
    system, user = render_prompt(VARIANT, obs(UP))
    with pytest.raises(ProviderError, match="unknown"):
        await ScriptedProvider("m", "oracle", COMMON, 0).complete(system, user)
    with pytest.raises(ProviderError, match="needs param"):
        await ScriptedProvider("m", "trend_chaser", {"chars_per_token": 4}, 0).complete(system, user)


@pytest.mark.parametrize(
    ("strategy", "params", "expected"),
    [
        ("buy_and_hold", {"weight": 0.5}, 0.5),
        ("ts_momentum", {"weight": 0.5, "lookback": 5}, 0.5),
        ("mean_reversion", {"weight": 0.5, "lookback": 10, "z_scale": 1.0}, -0.5),
    ],
)
async def test_baselines(strategy: str, params: dict[str, float], expected: float) -> None:
    agent = BaselineAgent(BaselineAgentConfig(id=strategy, strategy=strategy, params=params), seed=0)  # type: ignore[arg-type]
    decision = await agent.decide(obs(UP))
    assert not agent.is_llm
    assert agent.agent_id == strategy
    assert decision.targets["A"] == pytest.approx(expected)
    assert decision.usage is None


async def test_baseline_edge_cases() -> None:
    mom = BaselineAgent(BaselineAgentConfig(id="m", strategy="ts_momentum", params={"weight": 0.5, "lookback": 50}), 0)
    assert (await mom.decide(obs(UP))).targets["A"] == 0.0
    flat = BaselineAgent(
        BaselineAgentConfig(id="r", strategy="mean_reversion", params={"weight": 0.5, "lookback": 5, "z_scale": 1}), 0
    )
    assert (await flat.decide(obs([1.0] * 10))).targets["A"] == 0.0
    assert (await flat.decide(obs([1.0, 2.0]))).targets["A"] == 0.0
    rnd_a = BaselineAgent(BaselineAgentConfig(id="x", strategy="random", params={"weight": 0.5}), 3)
    rnd_b = BaselineAgent(BaselineAgentConfig(id="x", strategy="random", params={"weight": 0.5}), 3)
    assert (await rnd_a.decide(obs(UP))).targets == (await rnd_b.decide(obs(UP))).targets
    missing = BaselineAgent(BaselineAgentConfig(id="b", strategy="buy_and_hold", params={}), 0)
    with pytest.raises(ValueError, match="weight"):
        await missing.decide(obs(UP))
