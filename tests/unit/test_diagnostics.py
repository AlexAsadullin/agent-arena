from __future__ import annotations

import pytest

from agent_arena.diagnostics import diagnose, mean_turnover
from agent_arena.results import RunResult, StepRecord
from agent_arena.settings import DiagnosticsSettings

SETTINGS = DiagnosticsSettings(flat_epsilon=0.01, trade_epsilon=0.01)


def step(i: int, w_before: float, w_after: float, ret: float, target: float | None = None, **kw: object) -> StepRecord:
    return StepRecord(
        step=i,
        ts=f"2025-01-01T{i:02d}:00:00Z",
        equity=1000,
        equity_next=1000 * (1 + ret),
        ret_net=ret,
        ret_gross=ret,
        fees=0,
        token_cost=0,
        latency_cost=0,
        weights_before={"A": w_before},
        targets={"A": w_after if target is None else target},
        weights_after={"A": w_after},
        turnover=abs(w_after - w_before),
        traded_notional=0,
        n_orders=1,
        n_rejected=kw.get("rejected", 0),  # type: ignore[arg-type]
        n_clamped=0,
        reject_reasons=["capital"] * int(kw.get("rejected", 0)),  # type: ignore[call-overload]
        status=kw.get("status", "ok"),  # type: ignore[arg-type]
    )


def run(steps: list[StepRecord], seed: int = 0, variant: str | None = "v", episode: int = 0) -> RunResult:
    return RunResult(
        run_id=f"{seed}{variant}{episode}",
        season_id="s",
        agent_id="a",
        is_llm=True,
        model="m",
        variant=variant,
        episode=episode,
        seed=seed,
        session_id="x",
        started_at="",
        finished_at="",
        steps=steps,
    )


def test_martingale_pattern_is_detected() -> None:
    # Exposure doubles after every loss and resets after every gain.
    steps = [
        step(0, 0.0, 0.1, -0.01),
        step(1, 0.1, 0.2, -0.01),
        step(2, 0.2, 0.4, 0.02),
        step(3, 0.4, 0.1, -0.01),
        step(4, 0.1, 0.2, 0.01),
        step(5, 0.2, 0.1, 0.0),
    ]
    d = diagnose("a", [run(steps)], ["A"], baseline_turnover=0.05, settings=SETTINGS)
    assert d.martingale_score == pytest.approx(1.0)
    assert d.overtrading_ratio == pytest.approx(d.mean_turnover / 0.05)
    assert d.trades_per_run_max == 6
    assert d.concentration == pytest.approx(1.0)


def test_agreement_cv_limits_and_refusals() -> None:
    a = [step(0, 0, 0.5, 0.0), step(1, 0.5, 0.5, 0.0, status="refused")]
    b = [step(0, 0, 0.5, 0.0), step(1, 0.5, -0.5, 0.0, rejected=1)]
    c = [step(0, 0, 0.0, 0.0, status="error"), step(1, 0.0, 0.0, 0.0)]
    runs = [run(a, seed=0), run(b, seed=1), run(a, seed=0, variant="w")]
    d = diagnose("a", runs, ["A"], baseline_turnover=None, settings=SETTINGS)
    assert d.seed_agreement == pytest.approx(0.5)
    assert d.variant_agreement == pytest.approx(1.0)
    assert d.overtrading_ratio is None
    assert d.limit_hit_rate == pytest.approx(1 / 6)
    assert d.reject_reasons == {"capital": 1}
    assert d.refusal_rate == pytest.approx(2 / 6)
    assert d.trade_count_cv_across_seeds == pytest.approx(1 / 3)
    lone = diagnose("a", [run(c)], ["A"], None, SETTINGS)
    assert lone.seed_agreement is None
    assert lone.error_rate == pytest.approx(0.5)
    assert lone.martingale_score is None
    assert mean_turnover([]) == 0.0
    empty = diagnose("a", [], ["A"], None, SETTINGS)
    assert empty.n_runs == 0
