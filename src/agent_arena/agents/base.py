"""The single agent protocol shared by LLM agents and quant baselines.

Every agent sees the same :class:`Observation` (built by the runner from gateway tool calls) and
returns a :class:`Decision`: target portfolio weights per symbol (signed fraction of equity).
The runner turns targets into orders that go through the gateway's risk path, so LLM agents and
baselines compete under identical rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

DecisionStatus = Literal["ok", "refused", "error", "frozen"]


class Observation(BaseModel):
    season_id: str
    episode: int
    step: int
    seed: int
    known_at: datetime
    symbols: list[str]
    closes: dict[str, list[float]]
    features: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    forecasts: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float]
    equity: float
    cash: float
    max_weight: float
    token_budget_left_usd: float | None = None


class Usage(BaseModel):
    """What one LLM call consumed."""

    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class Decision(BaseModel):
    targets: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    status: DecisionStatus = "ok"
    error: str | None = None
    usage: Usage | None = None


class Agent(Protocol):
    """An arena competitor. ``decide`` must not mutate the observation."""

    @property
    def agent_id(self) -> str: ...

    @property
    def is_llm(self) -> bool: ...

    async def decide(self, obs: Observation) -> Decision: ...
