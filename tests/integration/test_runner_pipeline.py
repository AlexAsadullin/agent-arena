"""End-to-end: runner against the fake gateway, then the full pipeline and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from agent_arena.agents.factory import AgentFactory
from agent_arena.agents.llm import load_prompts
from agent_arena.cli import app
from agent_arena.gateway.fake import FakeGateway, fake_gateway_factory
from agent_arena.leaderboard import build_leaderboard
from agent_arena.pipeline import rebuild_leaderboard, run_season_pipeline
from agent_arena.results import RunResult
from agent_arena.runner import ArenaRunner
from agent_arena.season import SeasonConfig
from agent_arena.settings import ArenaSettings

FIXTURE = Path("tests/fixtures/tiny_season.yaml")


async def run(
    season: SeasonConfig,
    settings: ArenaSettings,
    http: httpx.AsyncClient,
    registry: dict[int, FakeGateway] | None = None,
) -> list[RunResult]:
    factory = fake_gateway_factory(season, settings.pricing, settings.gateway, registry)
    runner = ArenaRunner(season, settings, factory, AgentFactory(settings, load_prompts(settings.prompts_file), http))
    return await runner.run_season()


async def test_season_is_deterministic_and_charges_intelligence(
    season: SeasonConfig, settings: ArenaSettings, http: httpx.AsyncClient
) -> None:
    registry: dict[int, FakeGateway] = {}
    first = await run(season, settings, http, registry)
    second = await run(season, settings, http)
    n_llm = len(season.llm_agents) * len(season.prompt_variants) * len(season.seeds) * season.episodes
    n_base = len(season.baseline_agents) * len(season.seeds) * season.episodes
    assert len(first) == n_llm + n_base
    assert [r.net_returns() for r in first] == [r.net_returns() for r in second]
    for r in first:
        assert len(r.steps) == season.steps_per_episode
        intel = sum(s.token_cost + s.latency_cost for s in r.steps)
        if r.is_llm:
            assert intel > 0
            assert sum(r.gross_returns()) > sum(r.net_returns())
        else:
            assert intel == 0
            assert r.gross_returns() == r.net_returns()
    # Every session the runner opened is in the fake, and every order went through place_order.
    assert sum(len(f.sessions) for f in registry.values()) == len(first)
    fills = sum(1 for f in registry.values() for a in f.audit if a.decision == "fill")
    assert fills == sum(s.n_orders - s.n_rejected for r in first for s in r.steps)
    gambler = [r for r in first if r.agent_id == "llm-martingale-gambler"]
    assert any(s.status in ("refused", "error") for r in gambler for s in r.steps)


async def test_token_budget_breach_freezes_and_carries_book(
    season: SeasonConfig, settings: ArenaSettings, http: httpx.AsyncClient
) -> None:
    tight = season.model_copy(
        update={
            "budgets": season.budgets.model_copy(update={"max_token_cost_usd": 0.005}),
            "agents": [a for a in season.agents if a.id in ("llm-trend-chaser", "buy-and-hold")],
            "episodes": 1,
            "seeds": [0],
            "prompt_variants": ["neutral"],
        }
    )
    runs = await run(tight, settings, http)
    chaser = next(r for r in runs if r.agent_id == "llm-trend-chaser")
    assert chaser.frozen_reason == "token_cost"
    frozen = [s for s in chaser.steps if s.frozen]
    assert frozen
    assert all(s.status == "frozen" and s.n_orders == 0 for s in frozen[1:])
    assert len(chaser.steps) == tight.steps_per_episode
    baseline = next(r for r in runs if r.agent_id == "buy-and-hold")
    assert baseline.frozen_reason is None


async def test_pipeline_writes_artefacts_and_inbox(
    season: SeasonConfig, settings: ArenaSettings, http: httpx.AsyncClient
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.post(f"{settings.ledger.base_url}/v1/trials/batch").mock(side_effect=httpx.ConnectError("down"))
        mock.get(url__startswith=f"{settings.ledger.base_url}/v1/strategies/").mock(return_value=httpx.Response(404))
        mock.post(f"{settings.leakage.base_url}/v1/probe").mock(side_effect=httpx.ConnectError("down"))
        outcome = await run_season_pipeline(
            season, settings, fake_gateway_factory(season, settings.pricing, settings.gateway), http
        )
        board = outcome.leaderboard
        assert {r.agent_id for r in board.rows} == {a.id for a in season.agents}
        assert [r.rank for r in board.rows] == list(range(1, len(board.rows) + 1))
        assert all(r.dsr_source == "local-fallback" for r in board.rows)
        assert {r.leakage for r in board.rows if r.kind == "llm"} == {"not_audited"}
        assert {r.leakage for r in board.rows if r.kind == "baseline"} == {"n/a"}
        assert board.best_baseline in {a.id for a in season.baseline_agents}
        for row in board.rows:
            assert row.net_sharpe.ci_low <= row.net_sharpe.value <= row.net_sharpe.ci_high
            assert set(board.win_prob_bootstrap[row.agent_id]) == {r.agent_id for r in board.rows} - {row.agent_id}
        inbox = Path(outcome.trials_destination)
        lines = inbox.read_text().splitlines()
        assert len(lines) == len(outcome.runs)
        trial = json.loads(lines[0])
        assert trial["source"] == "s8"
        assert trial["schema_version"] == "1"
        assert len(trial["returns"]["values"]) == season.steps_per_episode
        for name in (
            "runs.jsonl",
            "steps.parquet",
            "meta.json",
            "leaderboard.json",
            "leaderboard.md",
            "leaderboard.html",
        ):
            assert (outcome.store.dir / name).exists()
        assert "Behavioural diagnostics" in (outcome.store.dir / "leaderboard.md").read_text()
        assert "<table>" in (outcome.store.dir / "leaderboard.html").read_text()
        again = await rebuild_leaderboard(season, settings, http)
        assert [r.agent_id for r in again.rows] == [r.agent_id for r in board.rows]


async def test_s6_verdict_and_s7_report_are_used(
    season: SeasonConfig, settings: ArenaSettings, http: httpx.AsyncClient
) -> None:
    runs = await run(season, settings, http)
    from agent_arena.clients.leakage import LeakageResult
    from agent_arena.clients.ledger import Verdict

    verdict = Verdict(strategy_id="x", n_trials=10, dsr=0.97, passed=True)
    leak = {a.id: LeakageResult(model_id=a.model, status="flagged", source="s7") for a in season.llm_agents}
    board = build_leaderboard(season, runs, settings, {"ts-momentum": verdict}, leak)
    row = next(r for r in board.rows if r.agent_id == "ts-momentum")
    assert (row.dsr, row.dsr_source, row.dsr_passed) == (0.97, "s6", True)
    assert {r.leakage for r in board.rows if r.kind == "llm"} == {"flagged"}


def test_cli_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_LEDGER__BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARENA_LEAKAGE__BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARENA_LEDGER__OFFLINE_INBOX", "false")
    monkeypatch.setenv("ARENA_STATS__BOOTSTRAP_SAMPLES", "200")
    cli = CliRunner()
    result = cli.invoke(app, ["run-season", "--season", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "net Sharpe" in result.output
    assert "trials logged to: dropped" in result.output
    lb = cli.invoke(app, ["leaderboard", "--season", str(FIXTURE)])
    assert lb.exit_code == 0, lb.output
    rep = cli.invoke(app, ["report", "--season", str(FIXTURE)])
    assert rep.exit_code == 0, rep.output
    assert "leaderboard.html" in rep.output
    assert (tmp_path / "data" / "agent-arena" / "seasons" / "tiny" / "leaderboard.md").exists()
