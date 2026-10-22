"""Statistical leaderboard: per-agent metrics with bootstrap CIs, DSR, pairwise win probabilities,
Bradley-Terry strengths, significance against the best baseline and behavioural diagnostics.

The primary metric is the annualised net Sharpe ratio of each run (net of fees, tokens and latency
slippage). Agents are ranked by its point estimate; the CI and significance columns say whether
the ranking can be trusted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

from agent_arena.clients.leakage import LeakageResult
from agent_arena.clients.ledger import Verdict
from agent_arena.diagnostics import AgentDiagnostics, diagnose, mean_turnover
from agent_arena.results import RunResult
from agent_arena.season import SeasonConfig
from agent_arena.settings import ArenaSettings
from agent_arena.stats.bootstrap import EpisodeDraws, FloatArray, bootstrap_means, percentile_ci, point_estimate
from agent_arena.stats.pairwise import (
    bootstrap_win_probability,
    bradley_terry,
    bt_win_probability,
    episode_win_matrix,
    holm_adjust,
    paired_bootstrap_pvalue,
)
from agent_arena.stats.sharpe import annualise, deflated_sharpe_ratio, sharpe_ratio
from agent_arena.timeutil import iso, utcnow

BPS = 10_000.0


class Estimate(BaseModel):
    value: float
    ci_low: float
    ci_high: float


class AgentRow(BaseModel):
    rank: int
    agent_id: str
    kind: Literal["llm", "baseline"]
    model: str | None
    n_runs: int
    n_episodes: int
    net_sharpe: Estimate
    gross_sharpe: Estimate
    net_return_pct: Estimate
    intelligence_cost_bps: Estimate
    token_cost_usd_per_run: float
    latency_cost_usd_per_run: float
    dsr: float | None
    dsr_source: Literal["s6", "local-fallback"]
    dsr_passed: bool | None
    p_vs_best_baseline: float | None
    p_holm: float | None
    significant: bool | None
    bt_strength: float
    leakage: str
    diagnostics: AgentDiagnostics


class Leaderboard(BaseModel):
    schema_version: str = "1"
    season_id: str
    computed_at: str
    metric: str
    ci_level: float
    bootstrap_samples: int
    alpha: float
    n_episodes: int
    n_trials: int
    best_baseline: str | None
    rows: list[AgentRow]
    win_prob_bootstrap: dict[str, dict[str, float]]
    win_prob_bradley_terry: dict[str, dict[str, float]]


def run_sharpe(run: RunResult, periods_per_year: float, *, gross: bool = False) -> float:
    returns = run.gross_returns() if gross else run.net_returns()
    return annualise(sharpe_ratio(returns), periods_per_year)


def run_total_return(run: RunResult) -> float:
    return float(np.prod([1.0 + r for r in run.net_returns()]) - 1.0)


def run_intel_cost_bps(run: RunResult, capital: float) -> float:
    return sum(s.token_cost + s.latency_cost for s in run.steps) / capital * BPS


def metric_matrix(runs: Sequence[RunResult], episodes: Sequence[int], values: Mapping[str, float]) -> FloatArray:
    """``(n_episodes, n_replicates)`` matrix of a per-run metric, NaN-padded."""
    by_episode: dict[int, list[float]] = defaultdict(list)
    for r in runs:
        by_episode[r.episode].append(values[r.run_id])
    width = max((len(v) for v in by_episode.values()), default=1)
    out = np.full((len(episodes), width), np.nan)
    for i, ep in enumerate(episodes):
        row = by_episode.get(ep, [])
        out[i, : len(row)] = row
    return out


def build_leaderboard(
    season: SeasonConfig,
    runs: Sequence[RunResult],
    settings: ArenaSettings,
    verdicts: Mapping[str, Verdict | None],
    leakage: Mapping[str, LeakageResult],
) -> Leaderboard:
    """Aggregate a season's runs. ``verdicts`` and ``leakage`` are keyed by agent id."""
    stats = settings.stats
    rng = np.random.default_rng(stats.seed)
    agent_ids = [a.id for a in season.agents if any(r.agent_id == a.id for r in runs)]
    episodes = sorted({r.episode for r in runs})
    by_agent = {a: [r for r in runs if r.agent_id == a] for a in agent_ids}
    ppy = season.periods_per_year

    net = {r.run_id: run_sharpe(r, ppy) for r in runs}
    gross = {r.run_id: run_sharpe(r, ppy, gross=True) for r in runs}
    total = {r.run_id: run_total_return(r) * 100.0 for r in runs}
    intel = {r.run_id: run_intel_cost_bps(r, season.capital) for r in runs}

    draws = EpisodeDraws.draw(len(episodes), stats.bootstrap_samples, rng)

    def estimate(agent: str, values: Mapping[str, float]) -> tuple[Estimate, FloatArray]:
        matrix = metric_matrix(by_agent[agent], episodes, values)
        boot = bootstrap_means(matrix, draws, rng)
        lo, hi = percentile_ci(boot, stats.ci_level)
        return Estimate(value=point_estimate(matrix), ci_low=lo, ci_high=hi), boot

    net_est = {a: estimate(a, net) for a in agent_ids}

    # Deflated Sharpe: one trial per agent configuration (LLM agent x prompt variant).
    config_returns: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        config_returns[r.config_key].extend(r.net_returns())
    trial_srs = [sharpe_ratio(v) for v in config_returns.values()]

    baselines = [a.id for a in season.baseline_agents if a.id in by_agent]
    best_baseline = max(baselines, key=lambda b: net_est[b][0].value, default=None)
    tested = [a for a in agent_ids if a != best_baseline]
    pvalues: dict[str, float] = {}
    if best_baseline is not None:
        # Paired: both bootstrap distributions use the same episode draws.
        for a in tested:
            diff_obs = net_est[a][0].value - net_est[best_baseline][0].value
            boot_diff = net_est[a][1] - net_est[best_baseline][1]
            pvalues[a] = paired_bootstrap_pvalue(diff_obs, boot_diff)
    holm = dict(zip(tested, holm_adjust([pvalues.get(a, float("nan")) for a in tested]), strict=True))

    episode_means = np.vstack([np.nanmean(metric_matrix(by_agent[a], episodes, net), axis=1) for a in agent_ids])
    strengths = bradley_terry(
        episode_win_matrix(episode_means),
        stats.bradley_terry_prior,
        stats.bradley_terry_max_iter,
        stats.bradley_terry_tol,
    )
    bt_prob = bt_win_probability(strengths)

    base_turnover = float(np.median([mean_turnover(by_agent[b]) for b in baselines])) if baselines else None
    llm_ids = {a.id for a in season.llm_agents}
    model_of = {a.id: a.model for a in season.llm_agents}

    rows: list[AgentRow] = []
    for idx, a in enumerate(agent_ids):
        verdict = verdicts.get(a)
        if verdict is not None and verdict.dsr is not None:
            dsr: float | None = verdict.dsr
            source: Literal["s6", "local-fallback"] = "s6"
            passed: bool | None = verdict.passed
        else:
            pooled = [x for r in by_agent[a] for x in r.net_returns()]
            local = deflated_sharpe_ratio(pooled, trial_srs)
            dsr = None if np.isnan(local) else local
            source = "local-fallback"
            passed = None if dsr is None else dsr >= stats.dsr_pass_threshold
        p = pvalues.get(a)
        ph = holm.get(a)
        runs_a = by_agent[a]
        rows.append(
            AgentRow(
                rank=0,
                agent_id=a,
                kind="llm" if a in llm_ids else "baseline",
                model=model_of.get(a),
                n_runs=len(runs_a),
                n_episodes=len({r.episode for r in runs_a}),
                net_sharpe=net_est[a][0],
                gross_sharpe=estimate(a, gross)[0],
                net_return_pct=estimate(a, total)[0],
                intelligence_cost_bps=estimate(a, intel)[0],
                token_cost_usd_per_run=float(np.mean([sum(s.token_cost for s in r.steps) for r in runs_a])),
                latency_cost_usd_per_run=float(np.mean([sum(s.latency_cost for s in r.steps) for r in runs_a])),
                dsr=dsr,
                dsr_source=source,
                dsr_passed=passed,
                p_vs_best_baseline=p,
                p_holm=ph,
                significant=None if ph is None or np.isnan(ph) else ph < stats.alpha,
                bt_strength=float(strengths[idx]),
                leakage=leakage[a].status if a in leakage else "n/a",
                diagnostics=diagnose(a, runs_a, season.symbols, base_turnover, settings.diagnostics),
            )
        )
    rows.sort(key=lambda r: r.net_sharpe.value, reverse=True)
    ranked = [row.model_copy(update={"rank": i + 1}) for i, row in enumerate(rows)]

    win_boot = {
        a: {b: bootstrap_win_probability(net_est[a][1], net_est[b][1]) for b in agent_ids if b != a} for a in agent_ids
    }
    win_bt = {a: {b: float(bt_prob[i, j]) for j, b in enumerate(agent_ids) if b != a} for i, a in enumerate(agent_ids)}
    return Leaderboard(
        season_id=season.season_id,
        computed_at=iso(utcnow()),
        metric="annualised net Sharpe ratio per run (net of fees, token cost and latency slippage)",
        ci_level=stats.ci_level,
        bootstrap_samples=stats.bootstrap_samples,
        alpha=stats.alpha,
        n_episodes=len(episodes),
        n_trials=len(config_returns),
        best_baseline=best_baseline,
        rows=ranked,
        win_prob_bootstrap=win_boot,
        win_prob_bradley_terry=win_bt,
    )
