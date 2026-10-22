"""Experiments behind docs/report.md (``make report``).

E1 - wrong-winner rate. Synthetic agents with known skill; compare three ranking protocols:
     a single run per agent ranked by PnL (the Alpha Arena / LiveTradeBench protocol), the mean
     over all runs without uncertainty, and the arena protocol (two-level bootstrap, a winner is
     declared only if its win probability against every rival reaches the threshold). Also the
     false-pass rate of PSR and DSR when no agent has skill.
E2 - sample size. Correct-call rate and CI width of the arena protocol for several designs
     (episodes x seeds x prompt variants).
E3 - net-of-intelligence-cost accounting on the stored synthetic season: gross versus net
     Sharpe, rank changes, and sweeps over account size and latency slippage rate.
E0 - variance components of run Sharpe in the stored season (episodes, seeds, prompt variants).

Tables go to ``$TRADEQ_DATA_DIR/agent-arena/reports/`` as Markdown and JSON.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import numpy as np
import typer
from numpy.typing import NDArray
from scipy.stats import kendalltau

from agent_arena.gateway.fake import fake_gateway_factory
from agent_arena.logging_setup import configure_logging
from agent_arena.pipeline import run_season_pipeline
from agent_arena.results import RunResult
from agent_arena.season import SeasonConfig, load_season
from agent_arena.settings import ArenaSettings, ExperimentSettings, load_settings
from agent_arena.stats.bootstrap import EpisodeDraws, bootstrap_means, percentile_ci
from agent_arena.stats.sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_ratio
from agent_arena.storage import SeasonStore

FloatArray = NDArray[np.float64]
BPS = 10_000.0
MS_PER_SECOND = 1000.0
MIN_EPISODES = 2
WILSON_Z = 1.959963984540054  # standard normal quantile for a 95% interval


# ------------------------------------------------------------------ helpers
def wilson(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float, float]:
    """Proportion with its Wilson score interval (Wilson 1927)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def fmt_rate(r: tuple[float, float, float]) -> str:
    return f"{100 * r[0]:.1f}% [{100 * r[1]:.1f}, {100 * r[2]:.1f}]"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


@dataclass(frozen=True)
class Design:
    episodes: int
    seeds: int
    variants: int

    @property
    def replicates(self) -> int:
        return self.seeds * self.variants

    def label(self) -> str:
        return f"{self.episodes}x{self.seeds}x{self.variants}"


# ------------------------------------------------------------------ synthetic agents with known skill
def simulate(cfg: ExperimentSettings, design: Design, skills: Sequence[float], rng: np.random.Generator) -> FloatArray:
    """Per-step returns, shape ``(K, episodes, replicates, steps)``, step noise sd 1.

    r = mu_k + beta_k * m[e, t] + u[k, e, rep] + eps, where mu_k is the agent's skill (per-step
    Sharpe against unit noise), m is a market factor shared by all agents in an episode, and u is a
    persistent offset of one seed x prompt-variant replicate.
    """
    k = len(skills)
    mu = np.asarray(skills, dtype=np.float64)[:, None, None, None] / math.sqrt(cfg.periods_per_year)
    beta = np.asarray(cfg.factor_betas[:k], dtype=np.float64)[:, None, None, None]
    market = rng.normal(0.0, cfg.factor_vol, (1, design.episodes, 1, cfg.steps_per_run))
    offset = rng.normal(0.0, cfg.replicate_offset_vol, (k, design.episodes, design.replicates, 1))
    noise = rng.standard_normal((k, design.episodes, design.replicates, cfg.steps_per_run))
    return np.asarray(mu + beta * market + offset + noise, dtype=np.float64)


def true_sharpes(cfg: ExperimentSettings, skills: Sequence[float]) -> FloatArray:
    """Population annualised Sharpe of each synthetic agent."""
    k = len(skills)
    mu = np.asarray(skills) / math.sqrt(cfg.periods_per_year)
    var = 1.0 + (np.asarray(cfg.factor_betas[:k]) * cfg.factor_vol) ** 2 + cfg.replicate_offset_vol**2
    return np.asarray(mu / np.sqrt(var) * math.sqrt(cfg.periods_per_year), dtype=np.float64)


def run_sharpes(returns: FloatArray, periods_per_year: float) -> FloatArray:
    """Annualised Sharpe of every run: shape ``(K, episodes, replicates)``."""
    sd = returns.std(axis=3, ddof=1)
    return np.asarray(returns.mean(axis=3) / sd * math.sqrt(periods_per_year), dtype=np.float64)


@dataclass(frozen=True)
class Calls:
    single_run: int
    mean_only: int
    protocol: int | None
    half_width_best: float


def rank_protocols(cfg: ExperimentSettings, returns: FloatArray, rng: np.random.Generator) -> Calls:
    sharpes = run_sharpes(returns, cfg.periods_per_year)
    single = int(np.argmax(returns[:, 0, 0, :].sum(axis=1)))
    means = sharpes.mean(axis=(1, 2))
    top = int(np.argmax(means))
    draws = EpisodeDraws.draw(sharpes.shape[1], cfg.bootstrap_samples, rng)
    boots = [bootstrap_means(sharpes[i], draws, rng) for i in range(sharpes.shape[0])]
    win = [float(np.mean(boots[top] > boots[j])) for j in range(len(boots)) if j != top]
    # With one episode the bootstrap has no between-episode spread to resample, so the arena abstains.
    enough = sharpes.shape[1] >= MIN_EPISODES
    declared = top if enough and min(win) >= cfg.win_probability_threshold else None
    lo, hi = percentile_ci(boots[0], 0.95)
    return Calls(single_run=single, mean_only=top, protocol=declared, half_width_best=(hi - lo) / 2)


def experiment_wrong_winner(cfg: ExperimentSettings) -> dict[str, Any]:
    """E1: how often each protocol names the wrong winner, with and without true skill differences."""
    rng = np.random.default_rng(cfg.seed)
    design = Design(cfg.episodes, cfg.seeds, cfg.variants)
    out: dict[str, Any] = {"design": design.label(), "replications": cfg.replications}
    for scenario, skills in (("skill", cfg.skill_sharpes), ("null", [0.0] * len(cfg.skill_sharpes))):
        best = int(np.argmax(true_sharpes(cfg, skills)))
        counts: defaultdict[str, int] = defaultdict(int)
        for _ in range(cfg.replications):
            calls = rank_protocols(cfg, simulate(cfg, design, skills, rng), rng)
            no_winner = scenario == "null"
            counts["single_wrong"] += no_winner or calls.single_run != best
            counts["mean_wrong"] += no_winner or calls.mean_only != best
            if calls.protocol is None:
                counts["protocol_abstain"] += 1
            elif no_winner or calls.protocol != best:
                counts["protocol_wrong"] += 1
            else:
                counts["protocol_correct"] += 1
        n = cfg.replications
        out[scenario] = {
            "true_annualised_sharpe": true_sharpes(cfg, skills).round(3).tolist(),
            "single_run_wrong": wilson(counts["single_wrong"], n),
            "mean_only_wrong": wilson(counts["mean_wrong"], n),
            "protocol_wrong": wilson(counts["protocol_wrong"], n),
            "protocol_abstain": wilson(counts["protocol_abstain"], n),
            "protocol_correct": wilson(counts["protocol_correct"], n),
        }
    out["deflation"] = experiment_deflation(cfg, design, rng)
    return out


def experiment_deflation(cfg: ExperimentSettings, design: Design, rng: np.random.Generator) -> dict[str, Any]:
    """False passes of PSR(0) >= 0.95 and DSR >= 0.95 among agent x variant configurations with no skill."""
    k = len(cfg.skill_sharpes)
    psr_pass = dsr_pass = total = 0
    for _ in range(cfg.replications):
        returns = simulate(cfg, design, [0.0] * k, rng)
        per_config: list[FloatArray] = []
        for agent in range(k):
            for v in range(cfg.variants):
                cols = [s * cfg.variants + v for s in range(cfg.seeds)]
                per_config.append(returns[agent][:, cols, :].reshape(-1))
        trial_srs = [sharpe_ratio(r) for r in per_config]
        for r in per_config:
            centred = r - r.mean()
            psr = probabilistic_sharpe_ratio(
                sharpe_ratio(r),
                r.size,
                float(np.mean(centred**3) / np.std(r) ** 3),
                float(np.mean(centred**4) / np.std(r) ** 4),
            )
            psr_pass += psr >= cfg.win_probability_threshold
            dsr_pass += deflated_sharpe_ratio(r.tolist(), trial_srs) >= cfg.win_probability_threshold
            total += 1
    return {
        "configs_per_season": k * cfg.variants,
        "configs_tested": total,
        "psr_false_pass": wilson(psr_pass, total),
        "dsr_false_pass": wilson(dsr_pass, total),
        "season_any_psr_pass_expected": 1 - (1 - psr_pass / total) ** (k * cfg.variants),
    }


def experiment_designs(cfg: ExperimentSettings) -> list[dict[str, Any]]:
    """E2: correct-call rate and CI half-width of the arena protocol per design."""
    rng = np.random.default_rng(cfg.seed + 1)
    best = int(np.argmax(true_sharpes(cfg, cfg.skill_sharpes)))
    rows = []
    for e, s, v in cfg.designs:
        design = Design(e, s, v)
        correct = wrong = 0
        widths = []
        single_right = 0
        for _ in range(cfg.replications):
            calls = rank_protocols(cfg, simulate(cfg, design, cfg.skill_sharpes, rng), rng)
            correct += calls.protocol == best
            wrong += calls.protocol is not None and calls.protocol != best
            single_right += calls.mean_only == best
            widths.append(calls.half_width_best)
        w = np.asarray(widths)
        rows.append(
            {
                "design": design.label(),
                "runs_per_agent": e * design.replicates,
                "correct_call": wilson(correct, cfg.replications),
                "wrong_call": wilson(wrong, cfg.replications),
                "mean_only_correct": wilson(single_right, cfg.replications),
                "ci_half_width": float(w.mean()),
                "ci_half_width_se": float(w.std(ddof=1) / math.sqrt(w.size)),
            }
        )
    return rows


# ------------------------------------------------------------------ stored season
def variance_components(season: SeasonConfig, runs: Sequence[RunResult]) -> list[dict[str, Any]]:
    """E0: spread of run Sharpe across episodes, across seeds and across prompt variants (LLM agents)."""
    ppy = season.periods_per_year
    out = []
    for agent in season.llm_agents:
        x: dict[tuple[int, int, str], float] = {}
        for r in runs:
            if r.agent_id == agent.id and r.variant is not None:
                x[(r.episode, r.seed, r.variant)] = sharpe_ratio(r.net_returns()) * math.sqrt(ppy)
        eps, seeds, variants = sorted({k[0] for k in x}), sorted({k[1] for k in x}), sorted({k[2] for k in x})
        cube = np.array([[[x[(e, s, v)] for v in variants] for s in seeds] for e in eps])
        out.append(
            {
                "agent": agent.id,
                "sd_between_episodes": float(cube.mean(axis=(1, 2)).std(ddof=1)),
                "sd_across_seeds": float(np.sqrt(cube.var(axis=1, ddof=1).mean())),
                "sd_across_variants": float(np.sqrt(cube.var(axis=2, ddof=1).mean())),
                "sd_total": float(cube.std(ddof=1)),
            }
        )
    return out


def _bootstrap_sharpe(
    season: SeasonConfig, runs: Sequence[RunResult], series: dict[str, list[float]], cfg: ExperimentSettings
) -> dict[str, tuple[float, float, float]]:
    """Per agent: mean run Sharpe of the given per-run series with a two-level bootstrap CI."""
    rng = np.random.default_rng(cfg.seed + 2)
    episodes = sorted({r.episode for r in runs})
    draws = EpisodeDraws.draw(len(episodes), cfg.bootstrap_samples, rng)
    out = {}
    for agent in season.agents:
        rows: dict[int, list[float]] = defaultdict(list)
        for r in runs:
            if r.agent_id == agent.id:
                rows[r.episode].append(sharpe_ratio(series[r.run_id]) * math.sqrt(season.periods_per_year))
        width = max(len(v) for v in rows.values())
        matrix = np.full((len(episodes), width), np.nan)
        for i, e in enumerate(episodes):
            matrix[i, : len(rows[e])] = rows[e]
        boot = bootstrap_means(matrix, draws, rng)
        lo, hi = percentile_ci(boot, 0.95)
        out[agent.id] = (float(np.nanmean(np.nanmean(matrix, axis=1))), lo, hi)
    return out


def costed_returns(run: RunResult, token_multiplier: float, latency_rate: float, max_bps: float) -> list[float]:
    """Net step returns recomputed for another token volume per decision and latency slippage rate.

    Multiplying token cost by m is equivalent to dividing the account size by m, because token cost
    is a fixed USD amount per decision while trading PnL scales with the account.
    """
    out = []
    for s in run.steps:
        latency_bps = min(max_bps, latency_rate * s.latency_ms / MS_PER_SECOND)
        latency = s.traded_notional * latency_bps / BPS
        out.append(s.ret_gross - (s.token_cost * token_multiplier + latency) / s.equity)
    return out


def experiment_costs(season: SeasonConfig, runs: Sequence[RunResult], settings: ArenaSettings) -> dict[str, Any]:
    """E3: what intelligence-cost accounting changes on the stored season."""
    cfg = settings.experiments
    gross = _bootstrap_sharpe(season, runs, {r.run_id: r.gross_returns() for r in runs}, cfg)
    net = _bootstrap_sharpe(season, runs, {r.run_id: r.net_returns() for r in runs}, cfg)
    ids = [a.id for a in season.agents]
    tau = kendalltau([gross[a][0] for a in ids], [net[a][0] for a in ids])
    cost_bps = {
        a: float(np.mean([sum(s.token_cost + s.latency_cost for s in r.steps) for r in runs if r.agent_id == a]))
        / season.capital
        * BPS
        for a in ids
    }
    token_share = {
        a: float(
            sum(s.token_cost for r in runs if r.agent_id == a for s in r.steps)
            / max(sum(s.token_cost + s.latency_cost for r in runs if r.agent_id == a for s in r.steps), 1e-12)
        )
        for a in ids
    }
    sweep = []
    for m in cfg.token_multipliers:
        for rate in cfg.latency_rates_bps_per_s:
            series = {r.run_id: costed_returns(r, m, rate, settings.latency.max_slippage_bps) for r in runs}
            res = _bootstrap_sharpe(season, runs, series, cfg)
            order = sorted(ids, key=lambda a: res[a][0], reverse=True)
            sweep.append(
                {
                    "token_multiplier": m,
                    "latency_bps_per_s": rate,
                    "net_sharpe": {a: res[a] for a in ids},
                    "rank": {a: order.index(a) + 1 for a in ids},
                }
            )
    return {
        "gross": gross,
        "net": net,
        "kendall_tau": float(tau.statistic),
        "kendall_p": float(tau.pvalue),
        "cost_bps_per_run": cost_bps,
        "token_share_of_cost": token_share,
        "sweep": sweep,
        "llm_agents": [a.id for a in season.llm_agents],
    }


# ------------------------------------------------------------------ rendering
def render(e0: list[dict[str, Any]], e1: dict[str, Any], e2: list[dict[str, Any]], e3: dict[str, Any]) -> str:
    parts = ["# Agent Arena report tables", "", "Generated by `make report`. Rates carry 95% Wilson intervals.", ""]
    parts += ["## E1 wrong-winner rate (design " + e1["design"] + f", {e1['replications']} replications)", ""]
    rows = []
    for sc in ("skill", "null"):
        d = e1[sc]
        rows.append(
            [
                sc,
                ", ".join(f"{x:.2f}" for x in d["true_annualised_sharpe"]),
                fmt_rate(d["single_run_wrong"]),
                fmt_rate(d["mean_only_wrong"]),
                fmt_rate(d["protocol_wrong"]),
                fmt_rate(d["protocol_abstain"]),
                fmt_rate(d["protocol_correct"]),
            ]
        )
    parts.append(
        md_table(
            [
                "scenario",
                "true Sharpe",
                "single run wrong",
                "mean only wrong",
                "arena wrong",
                "abstains",
                "arena correct",
            ],
            rows,
        )
    )
    d = e1["deflation"]
    parts += [
        "",
        "## E1b false passes without skill",
        "",
        md_table(
            ["configs tested", "PSR>=0.95", "DSR>=0.95", "P(any of a season's configs passes PSR)"],
            [
                [
                    str(d["configs_tested"]),
                    fmt_rate(d["psr_false_pass"]),
                    fmt_rate(d["dsr_false_pass"]),
                    f"{100 * d['season_any_psr_pass_expected']:.1f}%",
                ]
            ],
        ),
        "",
        "## E2 design (episodes x seeds x variants)",
        "",
        md_table(
            ["design", "runs/agent", "arena correct", "arena wrong", "mean-only correct", "CI half-width (Sharpe)"],
            [
                [
                    r["design"],
                    str(r["runs_per_agent"]),
                    fmt_rate(r["correct_call"]),
                    fmt_rate(r["wrong_call"]),
                    fmt_rate(r["mean_only_correct"]),
                    f"{r['ci_half_width']:.2f} (se {r['ci_half_width_se']:.2f})",
                ]
                for r in e2
            ],
        ),
        "",
        "## E0 variance components of run Sharpe (stored season, LLM agents)",
        "",
        md_table(
            ["agent", "sd between episodes", "sd across seeds", "sd across variants", "sd total"],
            [
                [
                    r["agent"],
                    f"{r['sd_between_episodes']:.2f}",
                    f"{r['sd_across_seeds']:.2f}",
                    f"{r['sd_across_variants']:.2f}",
                    f"{r['sd_total']:.2f}",
                ]
                for r in e0
            ],
        ),
        "",
        f"## E3 gross vs net (Kendall tau {e3['kendall_tau']:.2f}, p {e3['kendall_p']:.3f})",
        "",
    ]

    def est(t: Sequence[float]) -> str:
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"

    parts.append(
        md_table(
            ["agent", "gross Sharpe", "net Sharpe", "cost bps/run", "token share of cost"],
            [
                [
                    a,
                    est(e3["gross"][a]),
                    est(e3["net"][a]),
                    f"{e3['cost_bps_per_run'][a]:.1f}",
                    f"{100 * e3['token_share_of_cost'][a]:.2f}%",
                ]
                for a in e3["gross"]
            ],
        )
    )
    agents = list(e3["gross"])
    parts += [
        "",
        "## E3b sweep: net Sharpe (rank) by token volume per decision and latency slippage rate",
        "",
        md_table(
            ["token x", "bps/s", *agents],
            [
                [f"{s['token_multiplier']:g}", f"{s['latency_bps_per_s']:g}"]
                + [f"{s['net_sharpe'][a][0]:.1f} ({s['rank'][a]})" for a in agents]
                for s in e3["sweep"]
            ],
        ),
        "",
    ]
    return "\n".join(parts)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


async def _ensure_runs(season: SeasonConfig, settings: ArenaSettings) -> list[RunResult]:
    store = SeasonStore(settings.arena_dir, season.season_id)
    if store.runs_path.exists():
        return store.load_runs()
    async with httpx.AsyncClient() as http:
        factory = fake_gateway_factory(season, settings.pricing, settings.gateway)
        return (await run_season_pipeline(season, settings, factory, http)).runs


def main(season_path: Annotated[Path, typer.Option("--season", help="Season YAML for E0 and E3.")]) -> None:
    """Run E0-E3 and write the tables cited by docs/report.md."""
    settings = load_settings()
    configure_logging(settings.log_level)
    season = load_season(season_path)
    runs = asyncio.run(_ensure_runs(season, settings))
    cfg = settings.experiments
    e0 = variance_components(season, runs)
    e1 = experiment_wrong_winner(cfg)
    e2 = experiment_designs(cfg)
    e3 = experiment_costs(season, runs, settings)
    out = settings.arena_dir / cfg.reports_subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "report_tables.md").write_text(render(e0, e1, e2, e3), encoding="utf-8")
    payload = {"e0": e0, "e1": e1, "e2": e2, "e3": e3, "season_id": season.season_id}
    (out / "report_tables.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    typer.echo(render(e0, e1, e2, e3))
    typer.echo(f"written: {out / 'report_tables.md'}")


if __name__ == "__main__":
    typer.run(main)
