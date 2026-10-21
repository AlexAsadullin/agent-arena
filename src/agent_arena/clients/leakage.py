"""S7 leakage-probe client (contract 6.5).

For each LLM agent the arena asks S7 whether the model has memorised the season's instruments
over the season's window. Resolution order: the live ``POST /v1/probe`` endpoint, then a matching
report already written under ``$TRADEQ_DATA_DIR/llm-leakage-probe/reports/``, then the status
``not_audited``. A ``not_audited`` agent is shown as such on the leaderboard, never as clean.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_arena.settings import LeakageSettings
from agent_arena.timeutil import iso, parse_iso

log = structlog.get_logger(__name__)

LeakageStatus = Literal["clean", "flagged", "not_audited"]


class LeakageCell(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity: str
    date: str
    memorisation_score: float
    n_probes: int


class LeakageWindow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start: str
    end: str


class LeakageReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = "1"
    model_id: str
    window: LeakageWindow
    cells: list[LeakageCell] = Field(default_factory=list)
    flagged: bool
    threshold: float


class LeakageResult(BaseModel):
    model_id: str
    status: LeakageStatus
    source: Literal["s7", "s7-report-file", "none"]
    max_score: float | None = None
    threshold: float | None = None


def _result(report: LeakageReport, source: Literal["s7", "s7-report-file"]) -> LeakageResult:
    scores = [c.memorisation_score for c in report.cells]
    return LeakageResult(
        model_id=report.model_id,
        status="flagged" if report.flagged else "clean",
        source=source,
        max_score=max(scores) if scores else None,
        threshold=report.threshold,
    )


class LeakageClient:
    def __init__(self, settings: LeakageSettings, http: httpx.AsyncClient, data_root: Path) -> None:
        self.settings = settings
        self._http = http
        self._reports = data_root / settings.reports_subdir

    async def audit(self, model_id: str, entities: list[str], start: datetime, end: datetime) -> LeakageResult:
        body = {
            "model_id": model_id,
            "entities": entities,
            "start": iso(start),
            "end": iso(end),
            "languages": self.settings.languages,
        }
        try:
            resp = await self._http.post(
                f"{self.settings.base_url}/v1/probe", json=body, timeout=self.settings.timeout_s
            )
            if resp.status_code < 400:
                return _result(LeakageReport.model_validate(resp.json()), "s7")
            log.warning("leakage_probe_rejected", status=resp.status_code, model_id=model_id)
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            log.warning("leakage_probe_unavailable", model_id=model_id, error=str(exc))
        local = self._from_files(model_id, start, end)
        if local is not None:
            return local
        return LeakageResult(model_id=model_id, status="not_audited", source="none")

    def _from_files(self, model_id: str, start: datetime, end: datetime) -> LeakageResult | None:
        """The newest report file for ``model_id`` whose window covers the season."""
        if not self._reports.is_dir():
            return None
        for path in sorted(self._reports.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                report = LeakageReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
                covers = parse_iso(report.window.start) <= start and parse_iso(report.window.end) >= end
            except (ValueError, ValidationError, OSError):
                continue
            if report.model_id == model_id and covers:
                return _result(report, "s7-report-file")
        return None
