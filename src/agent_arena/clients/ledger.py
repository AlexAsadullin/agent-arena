"""S6 research ledger client (contract 6.1): trial capture and verdicts.

Every arena run becomes one ``TrialRecord``. When the ledger is unreachable, trials are appended
to the contract's offline inbox ``$TRADEQ_DATA_DIR/research-ledger/inbox/<source>-<yyyymmdd>.jsonl``
and verdicts are reported as unavailable, so the leaderboard falls back to its local DSR.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from agent_arena.results import RunResult
from agent_arena.settings import LedgerSettings
from agent_arena.timeutil import utcnow

log = structlog.get_logger(__name__)


class Actor(BaseModel):
    kind: Literal["human", "agent", "robot"]
    id: str


class ReturnsSeries(BaseModel):
    ts: list[str]
    values: list[float]


class TrialRecord(BaseModel):
    schema_version: str = "1"
    trial_id: str
    family: str
    strategy_id: str
    source: str
    actor: Actor
    params: dict[str, Any]
    dataset_fingerprint: str | None = None
    code_version: str | None = None
    seed: int | None
    started_at: str
    finished_at: str
    periods_per_year: float
    returns: ReturnsSeries
    metrics: dict[str, Any]
    tags: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = "1"
    strategy_id: str
    n_trials: int
    effective_trials: float | None = None
    sharpe: float | None = None
    psr: float | None = None
    dsr: float | None = None
    pbo: float | None = None
    min_trl: float | None = None
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    leakage_flags: list[Any] = Field(default_factory=list)
    computed_at: str | None = None


def trial_from_run(
    run: RunResult,
    strategy_id: str,
    family: str,
    source: str,
    periods_per_year: float,
    metrics: dict[str, Any],
    tags: list[str],
    code_version: str | None,
) -> TrialRecord:
    return TrialRecord(
        trial_id=str(uuid.uuid4()),
        family=family,
        strategy_id=strategy_id,
        source=source,
        actor=Actor(kind="agent" if run.is_llm else "robot", id=run.agent_id),
        params={
            "variant": run.variant,
            "model": run.model,
            "episode": run.episode,
            "session_id": run.session_id,
            "frozen_reason": run.frozen_reason,
        },
        code_version=code_version,
        seed=run.seed,
        started_at=run.started_at,
        finished_at=run.finished_at,
        periods_per_year=periods_per_year,
        returns=ReturnsSeries(ts=run.timestamps(), values=run.net_returns()),
        metrics=metrics,
        tags=tags,
    )


class LedgerClient:
    def __init__(self, settings: LedgerSettings, http: httpx.AsyncClient, data_root: Path) -> None:
        self.settings = settings
        self._http = http
        self._inbox = data_root / settings.inbox_subdir

    async def log_trials(self, trials: list[TrialRecord], now: datetime | None = None) -> str:
        """POST the batch; on failure append to the offline inbox. Returns ``"ledger"`` or the inbox path."""
        payload = [t.model_dump(mode="json") for t in trials]
        try:
            resp = await self._http.post(
                f"{self.settings.base_url}/v1/trials/batch", json=payload, timeout=self.settings.timeout_s
            )
            if resp.status_code < 400:
                return "ledger"
            log.warning("ledger_rejected", status=resp.status_code)
        except httpx.HTTPError as exc:
            log.warning("ledger_unreachable", error=str(exc))
        if not self.settings.offline_inbox:
            return "dropped"
        path = self._inbox / f"{self.settings.source}-{(now or utcnow()).strftime('%Y%m%d')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for trial in trials:
                fh.write(trial.model_dump_json() + "\n")
        return str(path)

    async def verdict(self, strategy_id: str) -> Verdict | None:
        try:
            resp = await self._http.get(
                f"{self.settings.base_url}/v1/strategies/{strategy_id}/verdict",
                timeout=self.settings.timeout_s,
            )
        except httpx.HTTPError:
            return None
        if resp.status_code >= 400:
            return None
        try:
            return Verdict.model_validate(resp.json())
        except ValueError:
            log.warning("verdict_malformed", strategy_id=strategy_id)
            return None
