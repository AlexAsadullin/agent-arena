"""Season pipeline: run -> store -> log trials to S6 -> leakage audit via S7 -> leaderboard -> reports."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

from agent_arena.agents.factory import AgentFactory
from agent_arena.agents.llm import load_prompts
from agent_arena.clients.leakage import LeakageClient, LeakageResult
from agent_arena.clients.ledger import LedgerClient, TrialRecord, Verdict, trial_from_run
from agent_arena.gateway.client import GatewayFactory
from agent_arena.leaderboard import Leaderboard, build_leaderboard, run_intel_cost_bps, run_sharpe, run_total_return
from agent_arena.report.render import render_html, render_markdown
from agent_arena.results import RunResult
from agent_arena.runner import ArenaRunner, strategy_id
from agent_arena.season import SeasonConfig
from agent_arena.settings import ArenaSettings
from agent_arena.storage import SeasonMeta, SeasonStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SeasonOutcome:
    runs: list[RunResult]
    leaderboard: Leaderboard
    store: SeasonStore
    trials_destination: str


def code_version(repo: Path) -> str | None:
    """Git sha of this repository, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_trials(season: SeasonConfig, runs: Sequence[RunResult], settings: ArenaSettings) -> list[TrialRecord]:
    version = code_version(Path.cwd())
    trials = []
    for run in runs:
        metrics = {
            "net_sharpe": run_sharpe(run, season.periods_per_year),
            "gross_sharpe": run_sharpe(run, season.periods_per_year, gross=True),
            "net_return": run_total_return(run),
            "intelligence_cost_bps": run_intel_cost_bps(run, season.budgets.capital),
            "token_cost_usd": sum(s.token_cost for s in run.steps),
            "latency_cost_usd": sum(s.latency_cost for s in run.steps),
            "turnover": sum(s.turnover for s in run.steps),
        }
        trials.append(
            trial_from_run(
                run,
                strategy_id(season.season_id, run.agent_id),
                f"agent-arena/{season.season_id}",
                settings.ledger.source,
                season.periods_per_year,
                metrics,
                ["agent-arena", season.season_id, "llm" if run.is_llm else "baseline"],
                version,
            )
        )
    return trials


async def audit_leakage(season: SeasonConfig, client: LeakageClient) -> dict[str, LeakageResult]:
    return {
        a.id: await client.audit(a.model, list(season.symbols), season.start, season.end) for a in season.llm_agents
    }


async def fetch_verdicts(season: SeasonConfig, ledger: LedgerClient) -> dict[str, Verdict | None]:
    return {a.id: await ledger.verdict(strategy_id(season.season_id, a.id)) for a in season.agents}


def publish(store: SeasonStore, board: Leaderboard) -> None:
    store.save_leaderboard(board, render_markdown(board), render_html(board))


async def run_season_pipeline(
    season: SeasonConfig,
    settings: ArenaSettings,
    gateway_factory: GatewayFactory,
    http: httpx.AsyncClient,
) -> SeasonOutcome:
    prompts = load_prompts(settings.prompts_file)
    runner = ArenaRunner(season, settings, gateway_factory, AgentFactory(settings, prompts, http))
    runs = await runner.run_season()
    store = SeasonStore(settings.arena_dir, season.season_id)
    store.save_runs(runs)

    ledger = LedgerClient(settings.ledger, http, settings.data_root)
    destination = await ledger.log_trials(build_trials(season, runs, settings))
    leakage = await audit_leakage(season, LeakageClient(settings.leakage, http, settings.data_root))
    store.save_meta(SeasonMeta(season_id=season.season_id, trials_destination=destination, leakage=leakage))

    board = build_leaderboard(season, runs, settings, await fetch_verdicts(season, ledger), leakage)
    publish(store, board)
    log.info("season_published", season=season.season_id, dir=str(store.dir), trials=destination)
    return SeasonOutcome(runs=runs, leaderboard=board, store=store, trials_destination=destination)


async def rebuild_leaderboard(season: SeasonConfig, settings: ArenaSettings, http: httpx.AsyncClient) -> Leaderboard:
    """Recompute the leaderboard from stored runs (fresh S6 verdicts, stored leakage results)."""
    store = SeasonStore(settings.arena_dir, season.season_id)
    runs = store.load_runs()
    meta = store.load_meta()
    ledger = LedgerClient(settings.ledger, http, settings.data_root)
    board = build_leaderboard(season, runs, settings, await fetch_verdicts(season, ledger), meta.leakage)
    publish(store, board)
    return board
