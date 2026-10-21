"""LLM agents: a prompt variant rendered from config, a provider call, and a strict answer parser.

Parsing rules: the answer must contain one JSON object with a ``targets`` mapping of symbol to a
finite weight. Unknown symbols are ignored and missing symbols keep their current weight. An
answer is a *refusal* when the provider flags it, when the JSON says ``"action": "refuse"``, or
when no JSON is present and the text matches a configured refusal pattern. Anything else that
does not parse is an *error*. On refusal or error the agent holds its current weights.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from string import Template
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_arena.agents.base import Decision, DecisionStatus, Observation, Usage
from agent_arena.agents.providers import Completion, LLMProvider, ProviderError
from agent_arena.timeutil import iso

# Closes are rounded before they enter the prompt: more digits cost tokens and carry no signal.
_PRICE_DIGITS = 6


class PromptVariant(BaseModel):
    system: str
    user: str


class PromptLibrary(BaseModel):
    variants: dict[str, PromptVariant] = Field(min_length=1)
    refusal_patterns: list[str] = Field(default_factory=list)

    def get(self, name: str) -> PromptVariant:
        try:
            return self.variants[name]
        except KeyError as exc:
            raise KeyError(f"prompt variant {name!r} is not defined") from exc


def load_prompts(path: Path) -> PromptLibrary:
    with path.open(encoding="utf-8") as fh:
        return PromptLibrary.model_validate(yaml.safe_load(fh))


def observation_payload(obs: Observation) -> dict[str, Any]:
    """The JSON object placed inside ``<observation>`` tags."""
    return {
        "known_at": iso(obs.known_at),
        "symbols": obs.symbols,
        "max_weight": obs.max_weight,
        "equity": round(obs.equity, 2),
        "weights": {s: round(w, 4) for s, w in obs.weights.items()},
        "closes": {s: [round(c, _PRICE_DIGITS) for c in v] for s, v in obs.closes.items()},
        "features": obs.features,
        "forecasts": obs.forecasts,
        "token_budget_left_usd": obs.token_budget_left_usd,
    }


def render_prompt(variant: PromptVariant, obs: Observation) -> tuple[str, str]:
    values = {
        "season_id": obs.season_id,
        "episode": str(obs.episode),
        "step": str(obs.step),
        "known_at": iso(obs.known_at),
        "symbols": ", ".join(obs.symbols),
        "max_weight": str(obs.max_weight),
        "equity": f"{obs.equity:.2f}",
        "observation_json": json.dumps(observation_payload(obs), separators=(",", ":")),
    }
    return Template(variant.system).substitute(values), Template(variant.user).substitute(values)


def _extract_json(text: str) -> Any | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_answer(
    completion: Completion, obs: Observation, refusal_patterns: list[str]
) -> tuple[DecisionStatus, dict[str, float], str, str | None]:
    """Return ``(status, targets, rationale, error)`` for a completion."""
    hold = dict(obs.weights)
    if completion.refused:
        return "refused", hold, "", "provider flagged a refusal"
    payload = _extract_json(completion.text)
    if payload is None:
        lowered = completion.text.lower()
        if any(p.lower() in lowered for p in refusal_patterns):
            return "refused", hold, "", "refusal text"
        return "error", hold, "", "no JSON object in answer"
    if not isinstance(payload, dict):
        return "error", hold, "", "answer JSON is not an object"
    if payload.get("action") == "refuse":
        return "refused", hold, str(payload.get("rationale", "")), "explicit refusal"
    raw = payload.get("targets")
    if not isinstance(raw, dict):
        return "error", hold, "", "answer has no 'targets' object"
    targets = dict(hold)
    for symbol, value in raw.items():
        if symbol not in obs.symbols:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            return "error", hold, "", f"non-numeric weight for {symbol}"
        targets[symbol] = float(value)
    return "ok", targets, str(payload.get("rationale", "")), None


class LLMAgent:
    def __init__(
        self,
        agent_id: str,
        provider: LLMProvider,
        variant: PromptVariant,
        refusal_patterns: list[str],
    ) -> None:
        self._agent_id = agent_id
        self._provider = provider
        self._variant = variant
        self._refusal_patterns = refusal_patterns

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def is_llm(self) -> bool:
        return True

    async def decide(self, obs: Observation) -> Decision:
        system, user = render_prompt(self._variant, obs)
        try:
            completion = await self._provider.complete(system, user)
        except ProviderError as exc:
            return Decision(targets=dict(obs.weights), status="error", error=str(exc))
        status, targets, rationale, error = parse_answer(completion, obs, self._refusal_patterns)
        usage = Usage(
            model=completion.model,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            latency_ms=completion.latency_ms,
        )
        return Decision(targets=targets, rationale=rationale, status=status, error=error, usage=usage)
