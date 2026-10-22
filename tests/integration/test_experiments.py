from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from agent_arena.agents.factory import AgentFactory
from agent_arena.agents.llm import load_prompts
from agent_arena.experiments import (
    Design,
    costed_returns,
    experiment_costs,
    experiment_designs,
    experiment_wrong_winner,
    main,
    simulate,
    true_sharpes,
    variance_components,
    wilson,
)
from agent_arena.gateway.fake import fake_gateway_factory
from agent_arena.runner import ArenaRunner
from agent_arena.season import SeasonConfig
from agent_arena.settings import ArenaSettings, ExperimentSettings

FIXTURE = Path("tests/fixtures/tiny_season.yaml")


def small(settings: ArenaSettings) -> ExperimentSettings:
    return settings.experiments.model_copy(
        update={"replications": 20, "bootstrap_samples": 100, "designs": [(1, 1, 1), (4, 2, 2)]}
    )


def test_wilson_reference() -> None:
    # Wilson (1927) interval for 8 successes out of 10 at 95%: about [0.490, 0.943].
    p, lo, hi = wilson(8, 10)
    assert p == 0.8
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_simulation_matches_true_sharpe(settings: ArenaSettings) -> None:
    import numpy as np

    cfg = settings.experiments.model_copy(update={"steps_per_run": 4000})
    rng = np.random.default_rng(0)
    returns = simulate(cfg, Design(40, 1, 1), cfg.skill_sharpes, rng)
    flat = returns.reshape(len(cfg.skill_sharpes), -1)
    measured = flat.mean(axis=1) / flat.std(axis=1) * np.sqrt(cfg.periods_per_year)
    assert measured == pytest.approx(true_sharpes(cfg, cfg.skill_sharpes), abs=1.0)


def test_protocols_on_small_budget(settings: ArenaSettings) -> None:
    cfg = small(settings)
    e1 = experiment_wrong_winner(cfg)
    assert e1["null"]["single_run_wrong"][0] == 1.0
    assert e1["null"]["protocol_wrong"][0] <= 0.2
    assert e1["deflation"]["dsr_false_pass"][0] <= e1["deflation"]["psr_false_pass"][0]
    e2 = experiment_designs(cfg)
    assert e2[0]["correct_call"][0] == 0.0  # one episode: the arena abstains
    assert e2[1]["ci_half_width"] > 0


async def test_cost_experiments_on_stored_runs(season: SeasonConfig, settings: ArenaSettings) -> None:
    async with httpx.AsyncClient() as http:
        factory = fake_gateway_factory(season, settings.pricing, settings.gateway)
        runs = await ArenaRunner(
            season, settings, factory, AgentFactory(settings, load_prompts(settings.prompts_file), http)
        ).run_season()
    llm_run = next(r for r in runs if r.is_llm)
    rebuilt = costed_returns(llm_run, 1.0, settings.latency.slippage_bps_per_second, settings.latency.max_slippage_bps)
    assert rebuilt == pytest.approx(llm_run.net_returns(), abs=1e-9)
    assert costed_returns(llm_run, 0.0, 0.0, 50.0) == pytest.approx(llm_run.gross_returns(), abs=1e-12)
    e3 = experiment_costs(season, runs, settings.model_copy(update={"experiments": small(settings)}))
    assert set(e3["net"]) == {a.id for a in season.agents}
    assert all(e3["net"][a][0] <= e3["gross"][a][0] + 1e-9 for a in e3["llm_agents"])
    e0 = variance_components(season, runs)
    assert {r["agent"] for r in e0} == {a.id for a in season.llm_agents}


def test_make_report_entry_point(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import typer

    monkeypatch.setenv("ARENA_EXPERIMENTS__REPLICATIONS", "10")
    monkeypatch.setenv("ARENA_EXPERIMENTS__BOOTSTRAP_SAMPLES", "100")
    monkeypatch.setenv("ARENA_EXPERIMENTS__DESIGNS", "[[2, 1, 1]]")
    monkeypatch.setenv("ARENA_LEDGER__BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARENA_LEAKAGE__BASE_URL", "http://127.0.0.1:9")
    app = typer.Typer()
    app.command()(main)
    result = CliRunner().invoke(app, ["--season", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "data" / "agent-arena" / "reports"
    assert (out / "report_tables.md").exists()
    assert (out / "report_tables.json").exists()
