"""Behavioural diagnostics computed from step records.

* Over-trading: mean per-step turnover ``sum_s |w_after - w_before|`` and its ratio to the median
  baseline turnover; trades per run (steps with turnover above ``trade_epsilon``) with its range
  and coefficient of variation across seeds under an identical prompt and market.
* Consistency: decision agreement - the share of run pairs whose target direction (long, flat,
  short) agrees on the same step and symbol - across seeds (same episode and prompt variant) and
  across prompt variants (same episode and seed).
* Sizing discipline: limit-hit rate (orders rejected by the gateway plus targets clamped to the
  weight cap, per step), concentration (Herfindahl index of absolute weights, 1/n = equal
  weight, 1 = one instrument), and a martingale score ``P(gross up | last step lost) - P(gross up
  | last step won)``, which is positive when an agent adds exposure after losses.
* Refusal and error rates over decision steps (frozen steps excluded).
"""

from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import BaseModel

from agent_arena.results import RunResult, StepRecord
from agent_arena.settings import DiagnosticsSettings


class AgentDiagnostics(BaseModel):
    agent_id: str
    mean_turnover: float
    overtrading_ratio: float | None
    trades_per_run_mean: float
    trades_per_run_min: int
    trades_per_run_max: int
    trade_count_cv_across_seeds: float | None
    seed_agreement: float | None
    variant_agreement: float | None
    limit_hit_rate: float
    reject_reasons: dict[str, int]
    concentration: float
    mean_gross_exposure: float
    martingale_score: float | None
    refusal_rate: float
    error_rate: float
    frozen_runs: int
    n_runs: int


def _direction(weight: float, eps: float) -> int:
    return 0 if abs(weight) <= eps else (1 if weight > 0 else -1)


def _agreement(group: Sequence[RunResult], symbols: Sequence[str], eps: float) -> float | None:
    """Mean pairwise direction agreement over runs that share a market path."""
    if len(group) < 2:
        return None
    n_steps = min(len(r.steps) for r in group)
    agree = total = 0
    for a, b in itertools.combinations(group, 2):
        for t in range(n_steps):
            for s in symbols:
                da = _direction(a.steps[t].targets.get(s, 0.0), eps)
                db = _direction(b.steps[t].targets.get(s, 0.0), eps)
                agree += da == db
                total += 1
    return agree / total if total else None


def _grouped_agreement(runs: Sequence[RunResult], key: str, symbols: Sequence[str], eps: float) -> float | None:
    groups: dict[tuple[int, object], list[RunResult]] = defaultdict(list)
    for r in runs:
        other = r.variant if key == "seed" else r.seed
        groups[(r.episode, other)].append(r)
    values = [v for g in groups.values() if (v := _agreement(g, symbols, eps)) is not None]
    return float(np.mean(values)) if values else None


def _gross(step: StepRecord) -> float:
    return sum(abs(w) for w in step.weights_after.values())


def _herfindahl(weights: Iterable[float]) -> float | None:
    absw = [abs(w) for w in weights]
    total = sum(absw)
    if total <= 0:
        return None
    return sum((w / total) ** 2 for w in absw)


def _martingale_score(runs: Sequence[RunResult], eps: float) -> float | None:
    up_after_loss = after_loss = up_after_gain = after_gain = 0
    for run in runs:
        for prev, cur in itertools.pairwise(run.steps):
            if cur.frozen or prev.ret_net == 0:
                continue
            went_up = _gross(cur) > _gross(prev) + eps
            if prev.ret_net < 0:
                after_loss += 1
                up_after_loss += went_up
            else:
                after_gain += 1
                up_after_gain += went_up
    if after_loss == 0 or after_gain == 0:
        return None
    return up_after_loss / after_loss - up_after_gain / after_gain


def _trades(run: RunResult, eps: float) -> int:
    return sum(1 for s in run.steps if s.turnover > eps)


def _seed_cv(runs: Sequence[RunResult], eps: float) -> float | None:
    groups: dict[tuple[int, str | None], list[int]] = defaultdict(list)
    for r in runs:
        groups[(r.episode, r.variant)].append(_trades(r, eps))
    cvs = [statistics.pstdev(c) / statistics.mean(c) for c in groups.values() if len(c) > 1 and statistics.mean(c) > 0]
    return float(np.mean(cvs)) if cvs else None


def diagnose(
    agent_id: str,
    runs: Sequence[RunResult],
    symbols: Sequence[str],
    baseline_turnover: float | None,
    settings: DiagnosticsSettings,
) -> AgentDiagnostics:
    steps = [s for r in runs for s in r.steps]
    live = [s for s in steps if not s.frozen]
    turnover = float(np.mean([s.turnover for s in steps])) if steps else 0.0
    trades = [_trades(r, settings.trade_epsilon) for r in runs]
    herf = [h for s in live if (h := _herfindahl(s.weights_after.values())) is not None]
    reasons: Counter[str] = Counter(reason for s in steps for reason in s.reject_reasons)
    n_live = max(len(live), 1)
    return AgentDiagnostics(
        agent_id=agent_id,
        mean_turnover=turnover,
        overtrading_ratio=turnover / baseline_turnover if baseline_turnover else None,
        trades_per_run_mean=float(np.mean(trades)) if trades else 0.0,
        trades_per_run_min=min(trades, default=0),
        trades_per_run_max=max(trades, default=0),
        trade_count_cv_across_seeds=_seed_cv(runs, settings.trade_epsilon),
        seed_agreement=_grouped_agreement(runs, "seed", symbols, settings.flat_epsilon),
        variant_agreement=_grouped_agreement(runs, "variant", symbols, settings.flat_epsilon),
        limit_hit_rate=sum(s.n_rejected + s.n_clamped for s in live) / n_live,
        reject_reasons=dict(sorted(reasons.items())),
        concentration=float(np.mean(herf)) if herf else 0.0,
        mean_gross_exposure=float(np.mean([_gross(s) for s in live])) if live else 0.0,
        martingale_score=_martingale_score(runs, settings.flat_epsilon),
        refusal_rate=sum(s.status == "refused" for s in live) / n_live,
        error_rate=sum(s.status == "error" for s in live) / n_live,
        frozen_runs=sum(r.frozen_reason is not None for r in runs),
        n_runs=len(runs),
    )


def mean_turnover(runs: Sequence[RunResult]) -> float:
    steps = [s.turnover for r in runs for s in r.steps]
    return float(np.mean(steps)) if steps else 0.0
