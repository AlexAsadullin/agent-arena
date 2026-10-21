"""Run and step records produced by the runner and consumed by statistics and reports."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_arena.agents.base import DecisionStatus


class StepRecord(BaseModel):
    """One decision step: the agent decides at ``ts`` and the result is marked one bar later.

    ``ret_net`` = (E_next - E_now) / E_now, after trading fees and intelligence cost.
    ``ret_gross`` adds the intelligence cost (tokens + latency) back: the return the same trades
    would have earned if thinking were free and instant.
    """

    step: int
    ts: str
    equity: float
    equity_next: float
    ret_net: float
    ret_gross: float
    fees: float
    token_cost: float
    latency_cost: float
    weights_before: dict[str, float]
    targets: dict[str, float]
    weights_after: dict[str, float]
    turnover: float
    traded_notional: float
    n_orders: int
    n_rejected: int
    n_clamped: int
    reject_reasons: list[str] = Field(default_factory=list)
    status: DecisionStatus
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    frozen: bool = False


class RunResult(BaseModel):
    run_id: str
    season_id: str
    agent_id: str
    is_llm: bool
    model: str | None
    variant: str | None
    episode: int
    seed: int
    session_id: str
    started_at: str
    finished_at: str
    frozen_reason: str | None = None
    steps: list[StepRecord]

    @property
    def config_key(self) -> str:
        """Agent configuration: an LLM agent under one prompt variant is one trial."""
        return self.agent_id if self.variant is None else f"{self.agent_id}[{self.variant}]"

    def net_returns(self) -> list[float]:
        return [s.ret_net for s in self.steps]

    def gross_returns(self) -> list[float]:
        return [s.ret_gross for s in self.steps]

    def timestamps(self) -> list[str]:
        return [s.ts for s in self.steps]
