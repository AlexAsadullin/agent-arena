from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from agent_arena.clients.leakage import LeakageClient
from agent_arena.clients.ledger import LedgerClient, TrialRecord, trial_from_run
from agent_arena.results import RunResult
from agent_arena.settings import ArenaSettings

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 2, 1, tzinfo=UTC)


def report(
    model: str, flagged: bool, start: str = "2024-12-01T00:00:00Z", end: str = "2025-03-01T00:00:00Z"
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "model_id": model,
        "window": {"start": start, "end": end},
        "cells": [{"entity": "BTC", "date": "2025-01-02", "memorisation_score": 0.8, "n_probes": 4}],
        "flagged": flagged,
        "threshold": 0.5,
    }


def a_run() -> RunResult:
    return RunResult.model_validate(
        {
            "run_id": "r1",
            "season_id": "s",
            "agent_id": "a",
            "is_llm": True,
            "model": "m",
            "variant": "neutral",
            "episode": 0,
            "seed": 1,
            "session_id": "x",
            "started_at": "2025-01-01T00:00:00Z",
            "finished_at": "2025-01-01T00:00:01Z",
            "steps": [],
        }
    )


@respx.mock
async def test_ledger_posts_batch_and_falls_back_to_inbox(settings: ArenaSettings, tmp_path: Path) -> None:
    trial = trial_from_run(a_run(), "agent-arena:s:a", "agent-arena/s", "s8", 8760, {"net_sharpe": 1.0}, ["t"], None)
    route = respx.post(f"{settings.ledger.base_url}/v1/trials/batch").mock(
        side_effect=[
            httpx.Response(200, json=[{"trial_id": trial.trial_id}]),
            httpx.Response(500),
            httpx.ConnectError("x"),
        ]
    )
    async with httpx.AsyncClient() as http:
        client = LedgerClient(settings.ledger, http, tmp_path)
        assert await client.log_trials([trial]) == "ledger"
        assert json.loads(route.calls[0].request.content)[0]["actor"] == {"kind": "agent", "id": "a"}
        day = datetime(2025, 5, 6, tzinfo=UTC)
        path = Path(await client.log_trials([trial], now=day))
        await client.log_trials([trial], now=day)
    assert path == tmp_path / "research-ledger" / "inbox" / "s8-20250506.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert TrialRecord.model_validate_json(lines[0]) == trial


@respx.mock
async def test_ledger_verdict(settings: ArenaSettings, tmp_path: Path) -> None:
    base = f"{settings.ledger.base_url}/v1/strategies"
    respx.get(f"{base}/ok/verdict").mock(
        return_value=httpx.Response(200, json={"strategy_id": "ok", "n_trials": 3, "dsr": 0.9, "passed": False})
    )
    respx.get(f"{base}/bad/verdict").mock(return_value=httpx.Response(200, json={"oops": 1}))
    respx.get(f"{base}/missing/verdict").mock(return_value=httpx.Response(404))
    respx.get(f"{base}/down/verdict").mock(side_effect=httpx.ConnectError("x"))
    async with httpx.AsyncClient() as http:
        client = LedgerClient(settings.ledger, http, tmp_path)
        verdict = await client.verdict("ok")
        assert verdict is not None
        assert verdict.dsr == 0.9
        for sid in ("bad", "missing", "down"):
            assert await client.verdict(sid) is None
        with pytest.raises(ValueError, match="'/'"):
            await client.verdict("agent-arena/s/a")


@respx.mock
async def test_leakage_probe_then_files_then_not_audited(settings: ArenaSettings, tmp_path: Path) -> None:
    route = respx.post(f"{settings.leakage.base_url}/v1/probe").mock(
        side_effect=[
            httpx.Response(200, json=report("m1", True)),
            httpx.ConnectError("x"),
            httpx.Response(503),
            httpx.ConnectError("x"),
        ]
    )
    reports = tmp_path / "llm-leakage-probe" / "reports"
    reports.mkdir(parents=True)
    (reports / "a.json").write_text(json.dumps(report("m2", False)))
    (reports / "b.json").write_text(json.dumps(report("m3", False, start="2025-01-15T00:00:00Z")))
    (reports / "c.json").write_text("{broken")
    async with httpx.AsyncClient() as http:
        client = LeakageClient(settings.leakage, http, tmp_path)
        live = await client.audit("m1", ["BTC"], START, END)
        assert (live.status, live.source, live.max_score) == ("flagged", "s7", 0.8)
        body = json.loads(route.calls[0].request.content)
        assert body["start"] == "2025-01-01T00:00:00Z"
        assert body["languages"] == settings.leakage.languages
        local = await client.audit("m2", ["BTC"], START, END)
        assert (local.status, local.source) == ("clean", "s7-report-file")
        uncovered = await client.audit("m3", ["BTC"], START, END)
        assert uncovered.status == "not_audited"
        none = await client.audit("m4", ["BTC"], START, END)
        assert (none.status, none.source) == ("not_audited", "none")
