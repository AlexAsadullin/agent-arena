from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent_arena.agents.base import Observation
from agent_arena.agents.llm import LLMAgent, PromptVariant, load_prompts, parse_answer, render_prompt
from agent_arena.agents.providers import Completion, ProviderError

PATTERNS = ["i cannot", "financial advice"]


def obs() -> Observation:
    return Observation(
        season_id="s",
        episode=0,
        step=3,
        seed=0,
        known_at=datetime(2025, 1, 1, tzinfo=UTC),
        symbols=["A", "B"],
        closes={"A": [1.0, 1.1], "B": [2.0, 2.1]},
        weights={"A": 0.2, "B": -0.1},
        equity=1000.0,
        cash=900.0,
        max_weight=1.0,
    )


def completion(text: str, refused: bool = False) -> Completion:
    return Completion(text=text, model="m", tokens_in=10, tokens_out=5, latency_ms=100.0, refused=refused)


def test_parse_ok_keeps_missing_and_ignores_unknown() -> None:
    text = 'Sure: {"targets": {"A": 0.5, "Z": 1}, "rationale": "up"}'
    status, targets, rationale, error = parse_answer(completion(text), obs(), PATTERNS)
    assert status == "ok"
    assert targets == {"A": 0.5, "B": -0.1}
    assert rationale == "up"
    assert error is None


@pytest.mark.parametrize(
    ("text", "refused", "status"),
    [
        ("I cannot provide financial advice.", False, "refused"),
        ('{"action": "refuse", "rationale": "no"}', False, "refused"),
        ('{"targets": {"A": 1}}', True, "refused"),
        ("buy everything", False, "error"),
        ("[1, 2]", False, "error"),
        ('{"targets": [1]}', False, "error"),
        ('{"targets": {"A": "big"}}', False, "error"),
        ('{"targets": {"A": true}}', False, "error"),
        ('{"targets": {"A": NaN}}', False, "error"),
        ("{not json}", False, "error"),
    ],
)
def test_parse_failures_hold_current_weights(text: str, refused: bool, status: str) -> None:
    got, targets, _, error = parse_answer(completion(text, refused), obs(), PATTERNS)
    assert got == status
    assert targets == {"A": 0.2, "B": -0.1}
    assert error


def test_render_prompt_embeds_observation() -> None:
    variant = PromptVariant(system="trade $symbols", user="<observation>$observation_json</observation> $max_weight")
    system, user = render_prompt(variant, obs())
    assert system == "trade A, B"
    payload = json.loads(user.split("<observation>")[1].split("</observation>")[0])
    assert payload["closes"]["A"] == [1.0, 1.1]
    assert user.endswith("1.0")


def test_config_prompts_load_and_render() -> None:
    library = load_prompts(__import__("pathlib").Path("config/prompts/variants.yaml"))
    for name in library.variants:
        system, user = render_prompt(library.get(name), obs())
        assert "<observation>" in user
        assert system
    with pytest.raises(KeyError):
        library.get("missing")


class _Provider:
    def __init__(self, result: Completion | Exception) -> None:
        self.result = result

    async def complete(self, system: str, user: str) -> Completion:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_agent_reports_usage_and_provider_errors() -> None:
    variant = PromptVariant(system="s", user="<observation>$observation_json</observation>")
    ok = LLMAgent("a", _Provider(completion('{"targets": {"A": 0.3}}')), variant, PATTERNS)
    decision = await ok.decide(obs())
    assert ok.is_llm
    assert ok.agent_id == "a"
    assert decision.status == "ok"
    assert decision.usage is not None
    assert decision.usage.tokens_in == 10
    failing = LLMAgent("a", _Provider(ProviderError("down")), variant, PATTERNS)
    bad = await failing.decide(obs())
    assert bad.status == "error"
    assert bad.usage is None
    assert bad.targets == {"A": 0.2, "B": -0.1}
