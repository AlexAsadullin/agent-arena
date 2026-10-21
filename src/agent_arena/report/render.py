"""Render a :class:`Leaderboard` as Markdown and as a static, self-contained HTML page."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, select_autoescape

from agent_arena.leaderboard import Estimate, Leaderboard


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or value != value:  # NaN check without importing math
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_est(est: Estimate, digits: int = 2) -> str:
    return f"{fmt(est.value, digits)} [{fmt(est.ci_low, digits)}, {fmt(est.ci_high, digits)}]"


def fmt_p(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.3f}"


def _yes_no(value: bool | None) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def render_markdown(board: Leaderboard) -> str:
    pct = round(board.ci_level * 100)
    lines = [
        f"# Agent Arena leaderboard: {board.season_id}",
        "",
        f"Metric: {board.metric}. Brackets are {pct}% two-level bootstrap confidence intervals "
        f"({board.bootstrap_samples} resamples over {board.n_episodes} episodes, then over seed x prompt "
        f"replicates). DSR deflates for {board.n_trials} agent configurations. p is the one-sided paired "
        f"bootstrap p-value against the best baseline ({board.best_baseline}), Holm-adjusted at alpha "
        f"{board.alpha}.",
        "",
        "| # | Agent | Kind | Net Sharpe | Gross Sharpe | Net return % | Intel. cost bps | DSR (source) | "
        "p vs best baseline (Holm) | Significant | BT strength | Leakage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in board.rows:
        lines.append(
            f"| {r.rank} | {r.agent_id} | {r.kind} | {fmt_est(r.net_sharpe)} | {fmt_est(r.gross_sharpe)} | "
            f"{fmt_est(r.net_return_pct)} | {fmt_est(r.intelligence_cost_bps, 1)} | "
            f"{fmt(r.dsr, 3)} ({r.dsr_source}) | {fmt_p(r.p_vs_best_baseline)} ({fmt_p(r.p_holm)}) | "
            f"{_yes_no(r.significant)} | {fmt(r.bt_strength)} | {r.leakage} |"
        )
    lines += [
        "",
        "## Behavioural diagnostics",
        "",
        "| Agent | Turnover/step | Over-trading x baseline | Trades/run (min-max) | Trade-count CV across seeds | "
        "Seed agreement | Variant agreement | Limit hits/step | Concentration | Martingale score | "
        "Refusal rate | Error rate | Frozen runs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in board.rows:
        d = r.diagnostics
        lines.append(
            f"| {r.agent_id} | {fmt(d.mean_turnover, 3)} | {fmt(d.overtrading_ratio)} | "
            f"{fmt(d.trades_per_run_mean, 1)} ({d.trades_per_run_min}-{d.trades_per_run_max}) | "
            f"{fmt(d.trade_count_cv_across_seeds)} | {fmt(d.seed_agreement)} | {fmt(d.variant_agreement)} | "
            f"{fmt(d.limit_hit_rate, 3)} | {fmt(d.concentration)} | {fmt(d.martingale_score)} | "
            f"{fmt(d.refusal_rate, 3)} | {fmt(d.error_rate, 3)} | {d.frozen_runs}/{d.n_runs} |"
        )
    ids = [r.agent_id for r in board.rows]
    lines += ["", "## Pairwise win probability P(row beats column), bootstrap", ""]
    lines += [_matrix(ids, board.win_prob_bootstrap)]
    lines += ["", "## Pairwise win probability P(row beats column), Bradley-Terry", ""]
    lines += [_matrix(ids, board.win_prob_bradley_terry)]
    lines.append("")
    return "\n".join(lines)


def _matrix(ids: list[str], probs: dict[str, dict[str, float]]) -> str:
    head = "| | " + " | ".join(ids) + " |"
    sep = "|---" * (len(ids) + 1) + "|"
    body = [f"| {a} | " + " | ".join("-" if a == b else fmt(probs[a].get(b)) for b in ids) + " |" for a in ids]
    return "\n".join([head, sep, *body])


def render_html(board: Leaderboard) -> str:
    env = Environment(
        loader=PackageLoader("agent_arena.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["fmt"] = fmt
    env.filters["est"] = fmt_est
    env.filters["p"] = fmt_p
    env.filters["yesno"] = _yes_no
    template = env.get_template("leaderboard.html.j2")
    return template.render(board=board, ids=[r.agent_id for r in board.rows], pct=round(board.ci_level * 100))


def summary_lines(board: Leaderboard) -> list[str]:
    """A short plain-text summary for the CLI."""
    out = [f"season {board.season_id}: {len(board.rows)} agents, {board.n_episodes} episodes"]
    for r in board.rows:
        sig = {True: "significant", False: "not significant", None: "-"}[r.significant]
        out.append(
            f"{r.rank:>2}. {r.agent_id:<26} net Sharpe {fmt_est(r.net_sharpe)}  "
            f"intel cost {fmt(r.intelligence_cost_bps.value, 1)} bps  vs best baseline: {sig}"
        )
    return out
